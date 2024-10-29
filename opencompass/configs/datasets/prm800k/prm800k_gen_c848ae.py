from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import PRM800kDataset, PRM800kEvaluator


EVAl_INIT_PROMPT = """You are a Math evaluator that evaluates the response step by step. Please follow the steps below:
1. There may be some errors in the solution steps. Please evaluate the response step by step and identify the first step that contains an error.
2. Read the original question and response carefully, and analysis the response based on the question step by step.
3. Identify the first step in the response if there is an error or give None if there is no error in the end of your evaluation.
4. The response may not finish the solution, like the response is only a part of the solution, please evaluate it as well.

Please reply strictly in the following format:
Detailed Step-by-step Analysis: (...Step-by-step detailed analysis of the response...)
First Error Step Number: (None if no error, e.g. None, otherwise the step number, e.g. Step 2)

Example evaluation response:
Example 1:
Detailed Step-by-step Analysis: ...
First Error Step Number: None

Example 2:
Detailed Step-by-step Analysis: ...
First Error Step Number: Step 2

[Original Question]: The original question that was asked.
{question}

[Response]: The original response.
{steps}

[Your Evaluation]: Your step-by-step evaluation of the response.
"""

prm800k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt=EVAl_INIT_PROMPT)
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))

prm800k_eval_cfg = dict(
    evaluator=dict(type=PRM800kEvaluator))

prm800k_datasets = [
    dict(
        type=PRM800kDataset,
        abbr='PRM800k',
        path='./data/prm800k/test.jsonl',
        reader_cfg=dict(
            input_columns=['question', 'steps'],
            output_column='answer',
        ),
        infer_cfg=prm800k_infer_cfg,
        eval_cfg=prm800k_eval_cfg)
]
