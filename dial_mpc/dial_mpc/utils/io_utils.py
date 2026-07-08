import importlib.resources
import os


def get_model_path(robot_name, model_name):
    with importlib.resources.path(f"dial_mpc.models.{robot_name}", model_name) as path:
        return path


def get_example_path(example_name):
    with importlib.resources.path(f"dial_mpc.examples", example_name) as path:
        return path


def get_repo_root():
    """Walk up from this file until a directory containing .git is found."""
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise RuntimeError(
                "Could not locate repo root (no .git directory found above dial_mpc package)"
            )
        path = parent


def resolve_output_dir(output_dir):
    """Anchor a relative output_dir to the repo root, independent of cwd."""
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(get_repo_root(), output_dir)


def load_dataclass_from_dict(dataclass, data_dict, convert_list_to_array=False):
    keys = dataclass.__dataclass_fields__.keys() & data_dict.keys()
    kwargs = {key: data_dict[key] for key in keys}
    if convert_list_to_array:
        import jax.numpy as jnp

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = jnp.array(value)
    return dataclass(**kwargs)
