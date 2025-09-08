from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import VerifierEvalDataset, VerifierEvaluator


QUESTION_QUALITY_PROMPT_EN_COT="""As a grading expert, your task is to determine whether the candidate's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct
   - The question is perfectly valid - never question them
   - Do not regenerate answers; only compare with the given standard

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure
     * Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components.
     * This determination must be made based on the question's phrasing and the nature of the standard answer.
   - Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors)
   - Disregard any differences in formatting or presentation style
   - For mathematical expressions: calculate step by step whether the two formulas are equivalent
   - For multiple-choice questions: compare only the final choice and corresponding option content

3. Multi-part Answers:
   - For questions requiring multiple responses (e.g., multi-select):
   - All parts must match the standard answer exactly. 
   - Compare each sub-answer step by step. Partial matches are considered incorrect.

4. Validity Check:
   - Reject answers that are:
     * Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) → Label as INCOMPLETE
     * Repetitive (repetition of words or phrases in a loop) → Label as ​REPETITIVE
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") → Label as ​REFUSAL
   - For invalid answers, specify the type in the judgment (e.g., \\boxed{C} - INCOMPLETE).


Grading Scale:
\\boxed{A} - CORRECT: 
   - Answer matches standard exactly (including equivalent expressions)
   - For numerical answers: consider as equivalent if values match when rounded appropriately
   - Semantically equivalent responses

\\boxed{B} - INCORRECT:
   - Any deviation from standard answer
   - Partial matches for multi-part questions

\\boxed{C} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL)

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate's answer including:
(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition), or a REFUSAL (explicit denial) - if so, immediately classify as \\boxed{C} with the corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order, multi-part completeness)
- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs, equivalent but unformatted expressions)
- Required answer type, precision level, etc.
(3) Perform a detailed comparison between the candidate's final answer and the standard answer, for example:
- Content equivalence
- Permitted variations in numerical precision
- Allowed expression formats]
Final Judgment: \\boxed{A/B/C} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{gold_answer}
<Standard Answer End>

<Candidate's Answer Begin>
{llm_response}
<Candidate's Answer End>

Analysis step by step and Final Judgment:
"""
ues_cot = True

# verifier_reader_cfg = dict(input_columns=['question, llm_response, gold_answer'], output_column='judgment_DeepSeek-v3_train_3class_only_final2_nothinking', test_split='test')
verifier_reader_cfg = dict(input_columns=['question, llm_response, gold_answer'], output_column='gold_judgment', test_split='test')


verifier_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt=QUESTION_QUALITY_PROMPT_EN_COT),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

verifier_eval_cfg = dict(
    evaluator=dict(type=VerifierEvaluator, two_label=True))

subsets = [
      #   'CompassAGI4S_v1.json'
      # 'CompassAGI4S_v1_b_sample.json'
      'CompassAGI4S_v1_b_sample_new.json'
      #   'verifier_vlm_v1.json'

        ]
verifier_datasets = []



for subset in subsets:
    verifier_datasets.append(
        dict(
            type=VerifierEvalDataset,
            abbr=f'verifier_{subset.split(".")[0]}_cot' if ues_cot else f'verifier_{subset.split(".")[0]}',
            # path='./data/verifier_eval',
            path='/fs-computility/llmeval/liuhongwei/work/main_work/opencompass/data/verifier_eval/',
            subset=subset,
            reader_cfg=verifier_reader_cfg,
            infer_cfg=verifier_infer_cfg,
            eval_cfg=verifier_eval_cfg)
    )
