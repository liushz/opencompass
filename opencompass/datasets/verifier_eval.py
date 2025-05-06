import json
import os.path as osp
import re

from datasets import Dataset

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
        dataset = Dataset.from_list(data)
        return dataset


def extract_last_boxed(response):
    pattern = r'\\boxed\{(.*?)\}'
    match = re.findall(pattern, response)
    try:
        if match:
            return match[-1]
        else:
            return None
    except Exception as e:
        print(f'Error extracting boxed content: {e}')
        return None


@ICL_EVALUATORS.register_module()
class VerifierEvaluator(BaseEvaluator):

    def __init__(self, two_label=True):
        self.two_label = two_label

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
            if self.two_label:
                is_correct = (pred == cand_ans or
                              (pred in ['c', 'b'] and cand_ans in ['b', 'c']))
            else:
                is_correct = (pred == cand_ans)
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

        score = cnt / len(predictions) * 100

        # 计算F1分数
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (
            precision + recall) > 0 else 0

        return {
            'score': score,
            'f1': f1 * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'fn ratio': fn / p_count,
            'fp ratio': fp / n_count,
            'details': details
        }
