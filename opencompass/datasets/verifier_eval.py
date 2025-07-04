import json
import os.path as osp
import re

from datasets import Dataset
from transformers import AutoTokenizer

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class VerifierEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, subset: str):
        file_path = osp.join(path, subset)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 计算每个样本的token数量
        # enc = tiktoken.encoding_for_model('cl100k_base')
        # enc = tiktoken.get_encoding('cl100k_base')
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
        for item in data:
            item['gold_answer'] = str(item['gold_answer'])
            if len(
                    tokenizer.encode(
                        str(item['question']) + str(item['llm_response']) +
                        str(item['gold_answer']))) + 2048 + 4096 > 32768:
                item['llm_response'] = item['llm_response'][-4096:]
        dataset = Dataset.from_list(data)
        return dataset


def extract_last_boxed(response):
    pattern_2 = r'\\boxed\{(.*?)\}'
    pattern_1 = r'\\boxed\{\{(.*?)\}\}'
    match_1 = re.findall(pattern_1, response)
    match_2 = re.findall(pattern_2, response)
    try:
        if match_1:
            return match_1[-1]
        elif match_2:
            return match_2[-1]
        else:
            return None
    except Exception as e:
        print(f'Error extracting boxed content: {e}')
        return None


def two_label_score(processed_predictions, references, ues_cot: bool = False):
    details = []
    cnt = 0
    tp = 0  # 真正例
    fp = 0  # 假正例
    fn = 0  # 假负例
    tn = 0  # 真负例
    p_count = 0
    n_count = 0

    for pred, cand_ans in zip(processed_predictions, references):
        if '</think>' in pred:
            pred = pred.split('</think>')[-1]
        pred, cand_ans = pred.strip().lower(), cand_ans.strip().lower()
        detail = {'pred': pred, 'answer': cand_ans, 'correct': False}
        # For xverify
        if pred == 'correct' or '[correct]' in pred or pred == 'yes':
            pred = 'a'
        elif pred == 'incorrect' or '[incorrect]' in pred or pred == 'no':
            pred = 'b'
        is_correct = (pred == cand_ans
                      or (pred in ['c', 'b'] and cand_ans in ['b', 'c']))
        cnt += int(is_correct)
        detail['correct'] = is_correct

        # 假设'a'为正类，'b'或'c'为负类
        if cand_ans == 'a':  # 实际为正类
            p_count += 1
            if pred == 'a':  # 预测为正类
                tp += 1
            else:  # 预测为负类
                fn += 1
        else:  # 实际为负类
            n_count += 1
            if pred == 'a':  # 预测为正类
                fp += 1
            else:  # 预测为负类
                tn += 1

        details.append(detail)

    score = cnt / len(processed_predictions) * 100

    # 计算F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision +
                                                             recall) > 0 else 0

    return {
        'score': score,
        'f1': f1 * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'fn count': fn,
        'fp count': fp,
        'details': details
    }


def three_label_score(processed_predictions, references):
    """Calculates scores for a 3-label classification task ('a', 'b', 'c').

    It computes:
    - Standard multi-class accuracy (prediction must exactly match ground truth).
    - Precision, Recall, F1-score for each class ('a', 'b', 'c') individually.
    - Macro-averaged F1-score.
    - A detailed list of predictions and their outcomes.

    Labels 'a', 'b', 'c' are treated as distinct classes for all metrics.
    Textual predictions like "correct", "yes" are normalized to 'a',
    and "incorrect", "no" are normalized to 'b'.
    """  # noqa: E501
    details_list = []
    standard_accuracy_correct_count = 0

    # Initialize TP, FP, FN for each class
    tp_a, fp_a, fn_a = 0, 0, 0
    tp_b, fp_b, fn_b = 0, 0, 0
    tp_c, fp_c, fn_c = 0, 0, 0

    for raw_pred_text, ground_truth_text in zip(processed_predictions,
                                                references):
        # 1. Pre-process raw prediction text
        processed_pred_text = raw_pred_text
        if '</think>' in processed_pred_text:
            processed_pred_text = processed_pred_text.split('</think>')[-1]
        processed_pred_text = processed_pred_text.strip().lower()

        # Pre-process ground truth text
        ground_truth_label = ground_truth_text.strip().lower()

        normalized_pred_label = processed_pred_text
        if processed_pred_text == 'correct' or '[correct]' in processed_pred_text or processed_pred_text == 'yes':  # noqa: E501 E261
            normalized_pred_label = 'a'
        elif processed_pred_text == 'incorrect' or '[incorrect]' in processed_pred_text or processed_pred_text == 'no':  # noqa: E501 E261
            normalized_pred_label = 'b'

        # 3. Calculate correctness for standard multi-class accuracy
        is_correct = (normalized_pred_label == ground_truth_label)
        if is_correct:
            standard_accuracy_correct_count += 1

        # 4. Store details for this instance
        instance_detail = {
            'raw_prediction': raw_pred_text,
            'processed_prediction_text': processed_pred_text,
            'normalized_prediction_label': normalized_pred_label,
            'ground_truth_label': ground_truth_label,
            'is_correct': is_correct  # Reflects standard accuracy
        }
        details_list.append(instance_detail)

        # Class 'a'
        if normalized_pred_label == 'a':
            if ground_truth_label == 'a':
                tp_a += 1
            else:
                fp_a += 1
        if ground_truth_label == 'a' and normalized_pred_label != 'a':
            fn_a += 1

        # Class 'b'
        if normalized_pred_label == 'b':
            if ground_truth_label == 'b':
                tp_b += 1
            else:
                fp_b += 1
        if ground_truth_label == 'b' and normalized_pred_label != 'b':
            fn_b += 1

        # Class 'c'
        if normalized_pred_label == 'c':
            if ground_truth_label == 'c':
                tp_c += 1
            else:
                fp_c += 1  # Predicted 'c' but was not 'c'
        # If ground truth is 'c'
        if ground_truth_label == 'c':
            if normalized_pred_label != 'c':  # Predicted not 'c' but was 'c'
                fn_c += 1

    # 6. Calculate standard multi-class accuracy score
    num_predictions = len(processed_predictions)
    accuracy_score = standard_accuracy_correct_count / num_predictions * 100 \
        if num_predictions > 0 else 0

    # 7. Helper function to calculate Precision, Recall, F1
    def calculate_prf1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (
            precision + recall) > 0 else 0
        return precision, recall, f1

    # Calculate P/R/F1 for each class
    precision_a, recall_a, f1_a = calculate_prf1(tp_a, fp_a, fn_a)
    precision_b, recall_b, f1_b = calculate_prf1(tp_b, fp_b, fn_b)
    precision_c, recall_c, f1_c = calculate_prf1(tp_c, fp_c, fn_c)

    # 8. Calculate Macro-F1
    macro_f1 = (f1_a + f1_b + f1_c) / 3

    # 9. Prepare and return results dictionary
    results = {
        'accuracy_score': accuracy_score,
        'macro_f1_score': macro_f1 * 100,
        'f1_a': f1_a * 100,
        'f1_b': f1_b * 100,
        'f1_c': f1_c * 100,
        'details': details_list
    }
    return results


@ICL_EVALUATORS.register_module()
class VerifierEvaluator(BaseEvaluator):

    def __init__(self, two_label=True, ues_cot=False):
        self.two_label = two_label
        self.ues_cot = ues_cot

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        processed_predictions = []

        # 从预测中提取答案
        for pred in predictions:
            if 'boxed' in pred:
                boxed_content = extract_last_boxed(pred)
                processed_predictions.append(
                    boxed_content if boxed_content else '')
            else:
                processed_predictions.append(pred)

        if self.two_label:
            return two_label_score(processed_predictions, references)
        else:
            return three_label_score(processed_predictions, references)
