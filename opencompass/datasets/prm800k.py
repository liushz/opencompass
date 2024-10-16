import json
import random
import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

random.seed(42)


@LOAD_DATASET.register_module()
class PRM800kDataset(BaseDataset):

    def parse_steps(self, steps):
        step_list = []
        bad_rate_idx = -1
        for idx, step in enumerate(steps):
            step_info = random.sample(step['completions'], 1)[0]
            step_list.append(step_info['text'])
            if step_info['rating'] == -1 and bad_rate_idx == -1:
                bad_rate_idx = idx + 1
        step_list = [
            f'Solution Step {idx + 1}: {step}'
            for idx, step in enumerate(step_list)
        ]
        step_list = '\n'.join(step_list)
        # bad_rate_idx = None if bad_rate_idx == -1 else bad_rate_idx
        return step_list, bad_rate_idx

    def transform(self, data):
        question = data['question']['problem']
        steps, bad_rate_idx = self.parse_steps(data['label']['steps'])
        answer = bad_rate_idx
        return {'question': question, 'steps': steps, 'answer': answer}

    def load(self, path: str):
        # path = get_data_path(path, local_mode=True)
        data = [json.loads(line) for line in open(path)]
        data = [
            item for item in data
            if item['label']['finish_reason'] in ['solution', 'found_error']
        ]
        dataset = Dataset.from_list(data)
        dataset = dataset.map(self.transform)
        return dataset


@ICL_EVALUATORS.register_module()
class PRM800kEvaluator(BaseEvaluator):

    def extract_step_num(self, input_str):
        # 使用正则表达式来匹配文本中的模式
        pattern = r'First Error Step Number:\s*(Step \d+|None)'
        match = re.search(pattern, input_str)

        if match:
            step_num = match.group(1)  # 返回匹配到的第一组内容
            if step_num == 'None':
                return None
            else:
                return int(step_num.split(' ')[1])  # 提取并转换成整数
        else:
            return None  # 如果没有匹配到，返回None

    def is_equal(self, pred, ref):
        if ref == -1 and pred in [None, 'None']:
            return True
        elif ref == pred:
            return True
        else:
            return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            i = self.extract_step_num(i)
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equal(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
