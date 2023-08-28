from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ToolBenchDataset, ToolBenchEvaluator

toolbench_reader_cfg = dict(
    input_columns=['query', 'avaliable_tools'],
    output_column='answer'
    )

toolbench_infer_cfg = dict(
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

toolbench_eval_cfg = dict(
    evaluator=dict(type=ToolBenchEvaluator), pred_postprocessor=dict(type='toolbench'))  # use the same processor to find answer

splits = ['G1_answer_converted','G2_answer_converted', 'G3_answer_converted']

toolbench_datasets = []

for split in splits:
    toolbench_datasets.append(
    dict(
        abbr='toolbench_{}'.format(split),
        type=ToolBenchDataset,
        path='./data/ToolBench',
        name = split,
        reader_cfg=toolbench_reader_cfg,
        infer_cfg=toolbench_infer_cfg,
        eval_cfg=toolbench_eval_cfg))