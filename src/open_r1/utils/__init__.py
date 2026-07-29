from .import_utils import is_e2b_available, is_morph_available

def get_model(*args, **kwargs):
    from .model_utils import get_model as _get_model

    return _get_model(*args, **kwargs)


def get_tokenizer(*args, **kwargs):
    from .model_utils import get_tokenizer as _get_tokenizer

    return _get_tokenizer(*args, **kwargs)


__all__ = ["get_tokenizer", "is_e2b_available", "is_morph_available", "get_model"]
