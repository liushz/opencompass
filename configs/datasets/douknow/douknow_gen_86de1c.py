from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets import DoUKnowDataset, douknow_postprocess
from opencompass.utils.text_postprocessors import first_capital_postprocess


single_choice_prompts = {
    "single_choice_cn": "以下是一道单项选择题，请你根据你了解的知识给出正确的答案选项。\n下面是你要回答的题目：\n{question}\n答案选项：",
}

douknow_sets = {
    'wiki': ['single_choice_cn'],
}

CircularEval = True

douknow_datasets = []

for _split in list(douknow_sets.keys()):
    for _name in douknow_sets[_split]:
        douknow_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>",
                    round=[
                        dict(
                            role="HUMAN",
                            prompt= single_choice_prompts[_name],
                        ),
                        dict(role="BOT", prompt="{answer}"),] if 'choice' in _name else cloze_prompts[_name],
                    ),
                ice_token="</E>",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )
        douknow_eval_cfg = dict(
            evaluator=dict(type=CircularEvaluator if CircularEval else AccEvaluator),
            pred_postprocessor=dict(type=first_capital_postprocess ) if 'single_choice' in _name else dict(type=douknow_postprocess, name=_name))

        douknow_datasets.append(
            dict(
                type=DoUKnowDataset,
                path=f"./data/douknow/{_split}/{_name}.jsonl",
                name='circular_' + _name if CircularEval else _name,
                abbr="douknow-" + _split + '-' + _name + 'circular'if CircularEval else '',
                reader_cfg=dict(
                    input_columns=["question"],
                    output_column="answer"
                    ),
                infer_cfg=douknow_infer_cfg,
                eval_cfg=douknow_eval_cfg,
            ))

del _split, _name