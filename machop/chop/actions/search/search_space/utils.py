def flatten_dict(nested_dict: dict, flattened, parent_key=None, separator="/") -> dict:
    """
    Flatten a nested dictionary

    ---
    Example
    ```python
    >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
    {"a/b": 1, "a/c": 2, "d": 3}
    ```

    """
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key is not None else key
        if isinstance(value, dict):  # If the value is another nested dictionary
            flatten_dict(value, flattened, new_key, separator=separator)
        else:  # If the value is a leaf node
            flattened[new_key] = value
    return flattened


def unflatten_dict(flat_dict: dict, separator="/") -> dict:
    """
    Unflatten a flattened dictionary

    ---
    Example
    ```python
    >>> unflatten_dict({"a/b": 1, "a/c": 2, "d": 3})
    {"a": {"b": 1, "c": 2}, "d": 3}
    ```
    """
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(separator)
        current_dict = nested_dict
        for i, k in enumerate(keys[:-1]):
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    return nested_dict
