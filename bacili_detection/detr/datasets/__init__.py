from .tb_bacillus import build

def build_dataset(image_set, args=None, **kwargs):
    return build(image_set, args, **kwargs)