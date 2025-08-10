from .data_simmim import build_loader_simmim as build_loader

# Backwards compatibility for callers passing is_pretrain
def build_loader_compat(config, logger, is_pretrain=True):
    return build_loader(config, logger)