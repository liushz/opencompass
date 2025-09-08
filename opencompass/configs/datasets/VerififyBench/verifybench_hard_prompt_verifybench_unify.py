from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import VerifierEvalDataset, VerifierEvaluator

VERIFYBENCH_PROMPT_COT = """
Given the following math problem and the reference answer. Judge the correctness of the answers given later, with some ability to generalize and match the form and format of the answer results.

The following specific requirements are followed when judging:
1. Judge only whether the final result of the reference answer and the answer to be judged agree; do not consider whether there are any errors in the process. Don't verify the correctness of the answer by yourself, please only refer to the reference answer for the correctness of the answer.
2. The reference answer and the answer to be judged only need to be essentially the same, ignoring irrelevant details such as units, symbols, whether or not to approximate, and the form of expression in the answer. The two answers are considered to be consistent if they are equivalently transformable.
3. All your analysis answer must be in English.
4. Please analyze the judged answer and try to compare it with the reference answer. At the end of all analysis, give the result of the judgment on an extra line at the end of the answer in the form 'Final Judgment: Yes/No'.

-------

Problem: {question}
Reference Answer: {gold_answer}
Solution to be evaluated: {llm_response}
"""

VERIFYBENCH_PROMPT_COT_UNIFY = """
<INPUT DATA BEGIN>:
{
'Problem': {question},
'Standard Answer': {gold_answer},
'Candidate's Answer': {llm_response}
}
<INPUT DATA END>

Given the following math problem and the reference answer. Judge the correctness of the answers given later, with some ability to generalize and match the form and format of the answer results.

The following specific requirements are followed when judging:
1. Judge only whether the final result of the reference answer and the answer to be judged agree; do not consider whether there are any errors in the process. Don't verify the correctness of the answer by yourself, please only refer to the reference answer for the correctness of the answer.
2. The reference answer and the answer to be judged only need to be essentially the same, ignoring irrelevant details such as units, symbols, whether or not to approximate, and the form of expression in the answer. The two answers are considered to be consistent if they are equivalently transformable.
3. All your analysis answer must be in English.
4. Please analyze the judged answer and try to compare it with the reference answer. At the end of all analysis, give the result of the judgment on an extra line at the end of the answer in the form 'Final Judgment: Yes/No'.

"""


PROMPT = VERIFYBENCH_PROMPT_COT_UNIFY  # COMPASSVERIFIER_PROMPT_COT # COMPASSVERIFIER_PROMPT_NO_COT
verifier_reader_cfg = dict(input_columns=['question, llm_response, gold_answer'], output_column='gold_judgment', test_split='test')

verifier_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt=PROMPT),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

verifier_eval_cfg = dict(
    evaluator=dict(
        type=VerifierEvaluator,
        two_label=True,
        yn_format=(PROMPT == VERIFYBENCH_PROMPT_COT)
    )
) 

subsets = ['verifybench_hard_v1.json']

verifier_datasets = []

for subset in subsets:
    verifier_datasets.append(
        dict(
            type=VerifierEvalDataset,
            abbr=f'verifier_{subset.split(".")[0]}_verifybench_unify',
            path='/mnt/shared-storage-user/liuhongwei/main_works/opencompass/data/verifier_eval/verifybench',
            subset=subset,
            reader_cfg=verifier_reader_cfg,
            infer_cfg=verifier_infer_cfg,
            eval_cfg=verifier_eval_cfg)
    )
