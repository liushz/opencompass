from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import VerifierEvalDataset, VerifierEvaluator

QUESTION_QUALITY_PROMPT_EN_COT="""
Please act as a grading expert. Your task is to evaluate the candidate's answer against the provided standard answer and classify it using the detailed system below.

**Evaluation Framework:**

**A: Correct** - The candidate's answer is fundamentally correct.
* **A1 - Identical:** The candidate's answer is almost identical to the standard answer in content, format, and wording.
* **A2 - Equivalent:** The answer is different in form or wording but is semantically, logically, or mathematically equivalent to the standard answer.
* **A3 - Substantially Correct:** The core conclusion is correct, but there are minor flaws or missing information that do not invalidate the overall answer.
* **A4 - Superset Correct:** The answer correctly provides all information from the standard answer and includes additional, relevant, and correct supplementary information.

**B: Incorrect** - The candidate's answer is incorrect.
* **B1 - Critically Incorrect:** The answer contains one or more critical factual, logical, or computational errors, rendering the entire answer invalid.
* **B2 - Partially Correct:** The answer contains some correct information but also has clear, critical errors.
* **B3 - Misinterprets Question:** The candidate seems to have misunderstood the question's intent, constraints, or scope, providing an answer to a different question.
* **B4 - Contradictory or Harmful:** The answer is the opposite of the truth or contains dangerous, misleading, or harmful information.

**C: Unjudgeable** - A fair comparison cannot be made.
* **C1 - Generation Quality Issue:** The candidate's output has technical problems (e.g., it is garbled, incomplete, cut off from the end, contains nonsensical repetition) that prevent a meaningful evaluation, just return C1.
* **C2 - Premise Issue:** The original question or the standard answer has an obvious quality problem (e.g., is ambiguous, incomplete, or mismatched with the question). This applies to flaws that are apparent without needing to solve the problem yourself.
* **C3 - Refusal:** The candidate explicitly refuses or states it is unable to provide an answer.

**Key Judging Principles:**
1.  **Final Answer Only:** Judge only the final answer provided. Completely ignore the reasoning process, even if it's flawed.
2.  **Multi-Part Questions:** For questions with multiple blanks or requiring multiple selections, the candidate's answer is only correct (A-category) if ALL parts match the standard answer exactly. Any partial match is incorrect (B-category).
3.  **Formatting:** Ignore formatting wrappers like `\\boxed{}` and judge the content inside.
4.  **Do Not Solve:** Your task is to compare the given answers, not to generate a new solution from scratch, because we can't judge whether your answer is correct or not.

Based on the criteria above, please assign the specific code that best describes the candidate's llm response within "\\boxed{}", like "\\boxed{A1}", in the end of your judgement.

<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>

Judge the candidate's answer and provide the classification code:
""".strip()


SYSTEM_PROMPT_SHORT = """A conversation between a User and an Assistant. The User poses a question, and the Assistant provides a solution. The Assistant's response follows these structured steps:

1. **Reasoning Process**: The Assistant comprehensively thinks about the problem through a reasoning process.
2. **Conclusion**: The Assistant reaches a conclusion. The final answer is highlighted within `\\boxed{...final answer...}`.
3. **Response Format**: The complete response should be formatted like:

...reasoning process...
...conclusion...
The answer is \\boxed{...final answer...}
""".strip()


verifier_reader_cfg = dict(input_columns=['question, llm_response, gold_answer'], output_column='gold_judgment', test_split='test')

verifier_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', prompt=SYSTEM_PROMPT_SHORT),
            ],
            round=[
                dict(role='HUMAN', prompt=QUESTION_QUALITY_PROMPT_EN_COT),
            ]
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

verifier_eval_cfg = dict(
    evaluator=dict(type=VerifierEvaluator, two_label=True))

subsets = [
        'Knowledge',
        'Math',
        'Science',
        'General Reasoning'
        ]
verifier_datasets = []

for subset in subsets:
    verifier_datasets.append(
        dict(
            type=VerifierEvalDataset,
            abbr=f'verifier_{subset.replace(" ", "_")}_sublabel',
            path='opencompass/VerifierBench',
            subset=subset,
            reader_cfg=verifier_reader_cfg,
            infer_cfg=verifier_infer_cfg,
            eval_cfg=verifier_eval_cfg)
    )
