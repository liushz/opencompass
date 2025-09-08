from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import VerifierEvalDataset, VerifierEvaluator

PROMPT= '''
Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer.  
The reference answer may be one single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.  
**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**  

Your task:  
- Compare the final output of the solution process with the reference answer.  
- If they **match exactly**, output **YES**.  
- If they **do not match**, output **NO**.  
- If the solution process is unclear, incomplete, or ambiguous, assume it is incorrect and output **NO**.  

Your output must be strictly **'YES'** or **'NO'**, with no additional words, punctuation, or explanation.  

---

**Question:**  
{question}  

**Solution Process (Final Step Only):**  
{llm_response}  

**Reference Answer:**  
{gold_answer}  

**Output:**  
'''

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
    evaluator=dict(type=VerifierEvaluator, two_label=True))

subsets = [
            'verifierbench_ood_v1_Math.json', 
           'verifierbench_ood_v1_Knowledge.json', 
           'verifierbench_ood_v1_Reasoning.json', 
           'verifierbench_id_v1_Math.json', 
           'verifierbench_id_v1_Knowledge.json', 
           'verifierbench_id_v1_Reasoning.json'
        'verifierbench_v3_General_Reasoning.json',
        'verifierbench_v3_Knowledge.json',
        'verifierbench_v3_Math.json',
        'verifierbench_v3_Science.json'
        # 'verifierbench_2819_Boolean.json',
        # 'verifierbench_2819_Multi-subproblem.json',
        # 'verifierbench_2819_Numerical.json',
        # 'verifierbench_2819_Short_Text.json',
        # 'verifierbench_2819_Formula.json',
        # 'verifierbench_2819_Multiple_Choice.json',
        # 'verifierbench_2819_Sequence.json'
           ]

verifier_datasets = []

for subset in subsets:
    verifier_datasets.append(
        dict(
            type=VerifierEvalDataset,
            abbr=f'verifier_{subset.split(".")[0]}',
            # path='./data/verifier_eval',
            path='/fs-computility/llmeval/liuhongwei/work/main_work/opencompass/data/verifier_eval/v2_all/0519',
            subset=subset,
            reader_cfg=verifier_reader_cfg,
            infer_cfg=verifier_infer_cfg,
            eval_cfg=verifier_eval_cfg)
    )
