import logging
import copy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import IncrementalNet, AKAIncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.loss import PES_Loss
from torchvision import transforms
from utils.toolkit import AutoencoderSigmoid
from utils.autoaugment import CIFAR10Policy
import time

EPSILON = 1e-8


class PRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        self._protos = {}
        if "cifar" in self.args["dataset"]:
            self.size = 32
        elif "tiny" in self.args["dataset"]:
            self.size = 56
        elif "imagenet" in self.args["dataset"]:
            self.size = 224
        self.pes_loss_func = PES_Loss()
        self.old_ae = None
    

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        self.save_checkpoint("checkpoint/{}/{}/{}/{}".format(self.args["model_name"],self.args["dataset"],self.args["init_cls"],self.args["increment"]))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        if self._cur_task == 1:
            self.old_ae = AutoencoderSigmoid(code_dims=512)
            self.old_ae.to(self._device)

        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes*4)
        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        
        resume = False
        if self._cur_task in []:
            self._network.load_state_dict(torch.load("checkpoint/{}/{}/{}/{}/phase{}.pkl".format(self.args["model_name"],self.args["dataset"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
            resume = True
            logging.info('!!!resume!!!')
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if not resume:
            if self._cur_task == 0:
                self._epoch_num = self.args["init_epochs"]
                optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["init_lr"], weight_decay=self.args["weight_decay"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=self.args["gamma"])
            else:
                trainable_list = nn.ModuleList([])
                trainable_list.append(self._network)
                trainable_list.append(self.old_ae)
                self._epoch_num = self.args["epochs"]
                logging.info('All params total: {}'.format(count_parameters(trainable_list)))
                optimizer = torch.optim.Adam(trainable_list.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])
            self._train_function(train_loader, test_loader, optimizer, scheduler)
        self._build_protos()
            
        
    def _build_protos(self):
        prototype = {}
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                prototype[class_idx] = class_mean
            self._protos.update(prototype)



    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            losses_new, losses_fkd, losses_proto, losses_pes, losses_pkd = 0., 0., 0., 0., 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                inputs = inputs.view(-1, 3, self.size, self.size)
                aug_targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
                logits, loss_new, loss_fkd, loss_proto, loss_pes, loss_pkd = self._compute_prl_loss(inputs, targets, aug_targets)
                loss = loss_new + loss_fkd + loss_proto + loss_pes + loss_pkd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_new += loss_new.item()
                losses_fkd += loss_fkd.item()
                losses_proto += loss_proto.item()
                losses_pes += loss_pes.item()
                losses_pkd += loss_pkd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aug_targets.expand_as(preds)).cpu().sum()
                total += len(aug_targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_iic {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_pkd {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_pes/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_pkd/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_iic {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_pkd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_pes/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_pkd/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)


    def _contras_loss(self, features, features_old):
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        features_old_norm = F.normalize(features_old, p=2, dim=1)
        protos = self._protos.values()
        protos = torch.from_numpy(np.asarray(list(protos))).float().to(self._device,non_blocking=True)
        protos = self.old_ae(protos)
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        similarity = similarity.sum() / (similarity.shape[0]*similarity.shape[1])
        return loss_align + similarity


    def _compute_prl_loss(self, inputs, targets, aug_targets):
        pes_targets = torch.stack([targets for k in range(4)], 1).view(-1)
        
        features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_clf = F.cross_entropy(logits/self.args["temp"], aug_targets)
        loss_new = loss_clf

        if self._cur_task == 0:
            loss_iic = self.args["lambda_pes"] * self.pes_loss_func(features, pes_targets)
            return logits, loss_new, torch.tensor(0.), torch.tensor(0.), loss_iic, torch.tensor(0.)
        
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_fkd = self.args["lambda_fkd"] * torch.dist(features, features_old, 2)
        loss_pkd = self.args["lambda_pgru"] * self._contras_loss(features, features_old)

        proto_features = []
        proto_targets = []
        old_class_list = list(self._protos.keys())
        for _ in range(features.shape[0]//4): # batch_size = feature.shape[0] // 4
            i = np.random.randint(0, features.shape[0])
            np.random.shuffle(old_class_list)
            lam = np.random.beta(0.5, 0.5)
            if lam > 0.6:
                lam = lam * 0.6
            if np.random.random() >= 0.5:
                temp = (1 + lam) * self._protos[old_class_list[0]] - lam * features.detach().cpu().numpy()[i]
            else:
                temp = (1 - lam) * self._protos[old_class_list[0]] + lam * features.detach().cpu().numpy()[i]
            proto_features.append(temp)
            proto_targets.append(old_class_list[0])

        proto_features = torch.from_numpy(np.asarray(proto_features)).float().to(self._device,non_blocking=True)
        proto_targets = torch.from_numpy(np.asarray(proto_targets)).to(self._device,non_blocking=True)
        
        proto_logits = self._network_module_ptr.fc(proto_features)["logits"]
        loss_proto = self.args["lambda_proto"] * F.cross_entropy(proto_logits/self.args["temp"], proto_targets*4)
                
        return logits, loss_new, loss_fkd, loss_proto, torch.tensor(0.), loss_pkd
        
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:, ::4]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        elif hasattr(self, '_protos'):
            protos = list(self._protos.values())
            y_pred, y_true = self._eval_nme(self.test_loader, protos/np.linalg.norm(protos,axis=1)[:,None])
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy