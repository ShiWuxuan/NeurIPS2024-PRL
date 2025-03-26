import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import sklearn
from torch.utils.data import DataLoader

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    checkpoint_dir = "checkpoint/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("average incremental accuracy: {}".format(np.mean(cnn_curve["top1"])))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("average incremental accuracy: {}".format(np.mean(nme_curve["top1"])))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("average incremental accuracy: {}".format(np.mean(cnn_curve["top1"])))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
        # if task == 10:
        #     test_dataset = data_manager.get_dataset(
        #         np.arange(0, 100), source='test', mode='test')
        #     test_loader = DataLoader(
        #         test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"])
        #     correct, total = 0.0, 0.0
        #     for i, (_, inputs, targets) in enumerate(test_loader):
        #         inputs = inputs.to(model._device)
        #         model._network_module_ptr.to(model._device)
        #         with torch.no_grad():
        #             outputs = model._network_module_ptr(inputs)["logits"]
        #         predicts = torch.max(outputs, dim=1)[1]
        #         if i == 0:
        #             total_predicts = predicts
        #             total_labels = targets
        #         else:
        #             total_predicts = torch.cat((total_predicts, predicts), dim=0)
        #             total_labels = torch.cat((total_labels, targets), dim=0)
        #         correct += (predicts.cpu() == targets.cpu()).sum()
        #         total += len(targets)
        #     accuracy = correct.item() / total
        #     print(accuracy)
        #     cm = sklearn.metrics.confusion_matrix(total_labels.cpu().numpy(), total_predicts.cpu().numpy())
        #     classes = list(range(0, 100))
        #     plot_confusion_matrix(cm, classes)

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    # tick_marks = np.arange(len(classes))[:]
    x_tick_marks = [0, 20, 40, 60, 80, 100]
    y_tick_marks = [0, 20, 40, 60, 80, 100]
    plt.xticks(x_tick_marks, x_tick_marks)
    plt.yticks(y_tick_marks, y_tick_marks)

    
    plt.tight_layout()
    plt.ylabel('True Classes', fontsize=21)
    plt.xlabel('Predicted Classes', fontsize=21)
    plt.savefig('confusion_matrix_PRL.svg', format='svg', bbox_inches = 'tight')
    plt.savefig('confusion_matrix_PRL.png', format='png', bbox_inches = 'tight')
    plt.show()


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
