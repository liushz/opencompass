from mmengine.config import read_base

# with read_base():
#     from .groups.agieval import agieval_summary_groups
#     from .groups.mmlu import mmlu_summary_groups
#     from .groups.ceval import ceval_summary_groups
#     from .groups.bbh import bbh_summary_groups
#     from .groups.GaokaoBench import GaokaoBench_summary_groups
#     from .groups.flores import flores_summary_groups
#     from .groups.jigsaw_multilingual import jigsaw_multilingual_summary_groups

summarizer = dict(
    dataset_abbrs=[
        ['douknow-wiki-single_choice_cncircular', 'acc_1'],
        ['douknow-wiki-single_choice_cncircular', 'perf_4'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
