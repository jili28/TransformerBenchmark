from src.models.DETRtime.model import build


def get_model(model_name: str, params: dict):
    if model_name == 'DETRtime':
        return build(params)
    else:
        raise ValueError("Choose a valid model.")
