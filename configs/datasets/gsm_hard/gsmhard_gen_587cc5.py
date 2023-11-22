from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSMHardDataset, gsm8k_postprocess, Gsm8kEvaluator

gsmhard_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsmhard_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin="</E>",
            round=[
                dict(role='HUMAN', prompt="Question: {question}\nAnswer:"),
                dict(role="BOT", prompt="{answer}"),
            ],
        ),                
        ice_token="</E>",
),

    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsmhard_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess))

gsmhard_datasets = [
    dict(
        abbr='gsm-hard',
        type=GSMHardDataset,
        path='./data/gsm-hard/test.jsonl',
        reader_cfg=gsmhard_reader_cfg,
        infer_cfg=gsmhard_infer_cfg,
        eval_cfg=gsmhard_eval_cfg)
]
