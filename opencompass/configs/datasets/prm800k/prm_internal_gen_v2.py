from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import PRM800kDatasetInternal, PRM800kEvaluator


EVAl_INIT_PROMPT_V3 = """You are a Math evaluator that evaluates the response step by step. Please follow the steps below:
You are a Math evaluator that evaluates the response step by step. Please follow the steps below:
1. There may be some errors in the solution steps. Please evaluate the response step by step and identify the first step that contains an error.
2. Read the original question and response carefully, and analysis the response based on the question step by step.
3. Identify the first step in the response if there is an error or give None if there is no error in the end of your evaluation.
4. The response may not finish the solution, like the response is only a part of the solution, please evaluate it as well.

Please reply strictly in the following format:
Detailed Analysis: (...Detailed analysis of the given response...)
First Error Step Number: (None if no error, e.g. None, otherwise the step number, e.g. Step 2)

[Original Question]:
{question}

[Original Response]:
{steps}

[Your Evaluation]: Your evaluation of the response.
"""

EVAl_INIT_PROMPT_V2 = """You are an Math evaluator that evaluates the response step by step. Please follow the steps below:

1. Read the original question and response carefully, and evaluate the response based on the original question step by step
2. Identify the first step in the response that contains an error and explain the error in detail. Write in which solution step number of the response the error first occured and then explain the error in the response detaily, how it impacts the response like a judgement to the response. like:
        First Error Step Number: ... e.g. Step 2
        Detailed Error Explanation: ...
3. If the response is correct every step, or there no serious error that affects the response, please provide a detailed explanation of why the response is correct or why the error is not serious, like:
        First Error Step Number: None
        Detailed Correct Explanation: ...

Please reply strictly in the following format:

First Error Step Number: ('None' if no error, otherwise the step number, e.g. 'Step 2')
Detailed Correct/Error Explanation: (...Combine your previous content with step-by-step detailed analysis...)

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
            dict(role='HUMAN', prompt=EVAl_INIT_PROMPT_V2)
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))

prm800k_eval_cfg = dict(
    evaluator=dict(type=PRM800kEvaluator))

prm_internal_gen_datasets = [
    dict(
        type=PRM800kDatasetInternal,
        abbr='PRMInternal',
        path='./data/prm_new/math50.json',
        reader_cfg=dict(
            input_columns=['question', 'steps'],
            output_column='answer',
        ),
        infer_cfg=prm800k_infer_cfg,
        eval_cfg=prm800k_eval_cfg)
]
