from opencompass.registry import TEXT_POSTPROCESSORS, LOAD_DATASET, ICL_EVALUATORS
from .base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator

import os
import json

from datasets import Dataset, load_dataset, DatasetDict


@TEXT_POSTPROCESSORS.register_module('toolbench_dataset')
def toolbench_dataset_postprocess(text: str) -> str:
    pass


@ICL_EVALUATORS.register_module()
class ToolBenchEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        pass
        

@LOAD_DATASET.register_module()
class ToolBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = []
        # splits = ['G1_answer_converted', 'G1_answer_converted', 'G1_answer_converted']
        if '.json' not in name:
            name += '.json'
        path = os.path.join(path, name)
        with open(path, 'r') as f:
            all_data = json.load(f)
            dataset = list(all_data.values())
        
        return Dataset.from_list(dataset)


