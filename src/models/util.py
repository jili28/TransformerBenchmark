from src.models.TransformerEncoder.model import build


def get_model(model_name: str, params: dict):
    if model_name == 'TransformerEncoder':
        return build(params)
    else:
        raise ValueError("Choose a valid model.")
