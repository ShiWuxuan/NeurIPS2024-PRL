from models.finetune import Finetune
from models.PRL import PRL

def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetune":
        return Finetune(args)
    elif name == "prl":
        return PRL(args)
    else:
        assert 0
