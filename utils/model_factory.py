"""
Return a model class based on the given string
"""

from src.models.TransformerEncoder.model import build
from src.models.SigmoidEncoder.model import build_encoder


def get_model(args, model_name: str = 'TransformerEncoder'):
    if model_name == 'TransformerEncoder':
        model, _  = build(args)
        return model
    elif model_name == 'Encoder':
        model = build_encoder(args)
        return model
    else:
        raise Exception("Choose valid model in config.py")
