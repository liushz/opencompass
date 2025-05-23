from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import VerifierEvalDataset, VerifierEvaluator

# QUESTION_QUALITY_PROMPT_EN = """Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
# Here are some evaluation criteria:
# 1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
# 2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
# 3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
# 4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
# 5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
# 6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), repetitive, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).
# Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
# A: CORRECT 
# B: INCORRECT
# C: INVALID
# After step by step judging, please output \\boxed{{A}} or \\boxed{{B}} or \\boxed{{C}} as the final judgement in the end.
# Here is your task. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
# <Original Question Begin>:
# {question}
# <Original Question End>
# <Gold Target Begin>:
# {gold_answer}
# <Gold Target End>
# <Predicted Answer Begin>:
# {llm_response}
# <Predicted Answer End>
# Judging the correctness of candidates' answers:
# """

PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{llm_response}"""

Correct answer: {gold_answer}

Judgement:
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
        #     'verifierbench_ood_v1_Math.json', 
        #    'verifierbench_ood_v1_Knowledge.json', 
        #    'verifierbench_ood_v1_Reasoning.json', 
        #    'verifierbench_id_v1_Math.json', 
        #    'verifierbench_id_v1_Knowledge.json', 
        #    'verifierbench_id_v1_Reasoning.json'
        'verifierbench_v3_General_Reasoning.json',
        'verifierbench_v3_Knowledge.json',
        'verifierbench_v3_Math.json',
        'verifierbench_v3_Science.json'
           ]

verifier_datasets = []

for subset in subsets:
    verifier_datasets.append(
        dict(
            type=VerifierEvalDataset,
            abbr=f'verifier_{subset.split(".")[0]}',
            # path='./data/verifier_eval',
            path='/fs-computility/llmeval/liuhongwei/work/main_work/opencompass/data/verifier_eval/v3',
            subset=subset,
            reader_cfg=verifier_reader_cfg,
            infer_cfg=verifier_infer_cfg,
            eval_cfg=verifier_eval_cfg)
    )
