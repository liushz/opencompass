import json
import random
import re

from datasets import Dataset, load_dataset

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

@LOAD_DATASET.register_module()
class PRM800kDatasetInternal(BaseDataset):

    def transform(self, data):
        question = data['query']
        steps = data['steps']
        steps = '\n'.join(['Solution Step'+ step for step in steps])
        if None not in data['step_gts']:
            answer = -1
        else:
            answer = data['step_gts'].index(None) + 1
        return {'question': question, 'steps': steps, 'answer': answer}

    def load(self, path: str):
        # path = get_data_path(path, local_mode=True)
        data = json.load(open(path))
        dataset = Dataset.from_list(data)
        dataset = dataset.map(self.transform)
        return dataset

def parse_steps(steps):
    # steps = steps[1:-1]

    # # 使用逗号分割字符串
    # list_steps = [item.strip() for item in steps.split("\', \'")]

    # # 去掉每个元素的引号
    # list_steps = [item.strip('\"') for item in list_steps]
    return eval(steps)
@LOAD_DATASET.register_module()
class PRM800kDatasetGSM(BaseDataset):

    def transform(self, data):
        question = data['question']
        steps = parse_steps(data['model_output_steps'])
        steps = '\n'.join(['Solution '+ step for step in steps])
        if not data['model_output_solution_first_error_step']:
            answer = -1
        else:
            answer = data['model_output_solution_first_error_step']
        return {'question': question, 'steps': steps, 'answer': int(answer)}

    def load(self, path: str):
        # path = get_data_path(path, local_mode=True)
        # 'Randolphzeng/Mr-GSM8K'
        data = [json.loads(line) for line in open(path)]
        dataset = Dataset.from_list(data)
        dataset = dataset.map(self.transform)
        return dataset


@ICL_EVALUATORS.register_module()
class PRM800kEvaluator(BaseEvaluator):

    def extract_step_num(self, input_str):
        pattern = r'First Error Step Number:\s*(Step \d+|None)'
        match = re.search(pattern, input_str)

        if match:
            step_num = match.group(1)
            if step_num == 'None':
                return None
            else:
                return int(step_num.split(' ')[1])
        else:
            return 'ERROR'

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
