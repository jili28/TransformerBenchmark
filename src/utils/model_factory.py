"""
Return a model class based on the given string
"""
from src.models.DETRtime.model import build
from src.models.SigmoidEncoder.model import build_encoder

def get_model(args, model_name: str = 'Acceptor'):
    if model_name == 'Acceptor':
        model  = build(args)
        return model
    elif model_name == 'Encoder':
        model = build_encoder(args)
        return model
    else:
        raise Exception("Choose valid model in config.py")
