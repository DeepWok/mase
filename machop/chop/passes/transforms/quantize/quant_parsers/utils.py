from copy import deepcopy


def cp_multi_values(
    src: dict, dst: dict, src_keys: tuple, dst_keys: tuple = None, strict: bool = True
):
    """Copy multiple values from src dict to dst dict."""
    if dst_keys is None:
        for key in src_keys:
            if not strict and key not in src:
                continue
            dst[key] = deepcopy(src[key])
    else:
        for src_key, dst_key in zip(src_keys, dst_keys):
            if not strict and src_key not in src:
                continue
            dst[dst_key] = deepcopy(src[src_key])


def has_multi_keys(src: dict, keys: tuple):
    """Check if src dict has multiple keys."""
    for key in keys:
        if key not in src:
            return False
    return True
