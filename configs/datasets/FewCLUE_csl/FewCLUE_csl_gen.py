from mmengine.config import read_base

with read_base():
    from .FewCLUE_csl_gen_1b0c02 import csl_datasets  # noqa: F401, F403
