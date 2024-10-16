from mmengine.config import read_base

with read_base():
    from .prm800k_gen_2f126c import prm800k_datasets  # noqa: F401, F403
