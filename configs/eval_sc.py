from mmengine.config import read_base
from transformers import AutoModelForCausalLM, AutoTokenizer


with read_base():
    # from .aliyun_env import llm_infer as infer, llm_eval as eval 

    # specify
    # from .datasets.MathBench.mathbench_gen import mathbench_datasets
    from .datasets.douknow.douknow_gen import douknow_datasets

    # from .datasets.XLSum.XLSum_gen_2bb71c import XLSum_datasets
    # from .datasets.cmmlu.cmmlu_ppl import cmmlu_datasets
    # from .datasets.xiezhi.xiezhi_ppl_ea6bd7 import xiezhi_datasets
    # from .datasets.tydiqa.tydiqa_gen_978d2a import tydiqa_datasets
    # from .datasets.squad20.squad20_gen_1710bc import squad20_datasets
    # from .datasets.drop.drop_gen_599f07 import drop_datasets
    # from .datasets.XCOPA.XCOPA_ppl_54058d import XCOPA_datasets
    # from .datasets.anli.anli_ppl_1d290e import anli_datasets
    # from .datasets.collections.base_medium import datasets
    # from .datasets.collections.base_medium_llama import datasets
    # from .datasets.gsm8k.gsm8k_gen_a3e34a import gsm8k_datasets
    # from .datasets.bbh.bbh_gen_6bd693 import bbh_datasets
    # from .datasets.commonsenseqa.commonsenseqa_ppl_5545e2 import commonsenseqa_datasets

    # from .models.qwen.hf_qwen_14b import models as qwen_14b_model
    # from .models.qwen.hf_qwen_14b_chat import models as qwen_14b_chat_model
    # from .models.qwen.hf_qwen_7b import models as qwen_7b_model
    from .models.qwen.hf_qwen_7b_chat import models as qwen_7b_chat_model
    # # from .models.baichuan.hf_baichuan2_13b_base import models as baichuan2_13b_base_model
    # from .models.baichuan.hf_baichuan2_13b_chat import models as baichuan2_13b_chat_model
    # from .models.baichuan.hf_baichuan2_7b_chat import models as baichuan2_7b_chat_model
    # from .models.baichuan.hf_baichuan2_7b_base import models as baichuan2_7b_base_model
    # from .models.hf_internlm.hf_internlm_7b import models as internlm_7b_model
    # from .models.hf_internlm.hf_internlm_chat_7b import models as internlm_chat_7b_model
    # from .models.llama.llama2_7b import models as llama2_7b_model
    # from .models.llama.llama2_13b import models as llama2_13b_model
    # from .models.llama.llama2_70b import models as llama2_30b_model

    # from .models.hf_moss_moon_003_base import models as moss_moon_003_base_model
    # from .models.hf_moss_moon_003_sft import models as moss_moon_003_sft_model
    # from .models.hf_mpt_7b import models as mpt_7b_model
    # from .models.hf_mpt_instruct_7b import models as mpt_instruct_7b_model
    # from .models.hf_tigerbot_7b_base import models as tigerbot_7b_base_model
    # from .models.hf_tigerbot_7b_sft import models as tigerbot_7b_sft_model
    # from .models.hf_wizardlm_7b import models as wizardlm_7b_model
    # # from .models.hf_wizardlm_70b import models as wizardlm_70b_model
    # from .models.hf_baichuan_13b_chat import models as baichuan_13b_chat_model
    # from .models.hf_baichuan_13b_base import models as baichuan_13b_base_model
    # from .models.hf_internlm_chat_7b_8k import models as internlm_chat_7b_8k_model
    # from .models.hf_vicuna_7b import models as vicuna_7b_model
    # from .models.hf_vicuna_13b import models as vicuna_13b_model
    # from .models.hf_vicuna_33b import models as vicuna_33b_model
    # from .models.hf_llama2_chinese_7b import models as llama2_chinese_7b_model
    # from .models.hf_llama2_chinese_13b import models as llama2_chinese_13b_model
    # from .models.hf_freewilly2_70b import models as freewilly2_70b_model
    # from .models.alpaca_7b import models as alpaca_7b_model
    # from .models.gogpt_7b import models as gogpt_7b_model
    # from .models.origin_llama_7b import models as origin_llama_7b_model
    # from .models.origin_llama_13b import models as origin_llama_13b_model
    # from .models.origin_llama_30b import models as origin_llama_30b_model
    # from .models.origin_llama_65b import models as origin_llama_65b_model
    # from .models.origin_llama2_7b import models as origin_llama2_7b_model
    # from .models.origin_llama2_13b import models as origin_llama2_13b_model
    # from .models.origin_llama2_70b import models as origin_llama2_70b_model
    # from .models.origin_llama2_7b_chat_cab import models as origin_llama2_7b_chat_cab_model
    # from .models.origin_llama2_13b_chat_cab import models as origin_llama2_13b_chat_cab_model
    # from .models.origin_llama2_70b_chat_cab import models as origin_llama2_70b_chat_cab_model
    # from .models.hf_qwen_7b import models as qwen_7b_model
    # from .models.hf_qwen_7b_chat import models as qwen_7b_chat_model
    # from .models.hf_chinese_llama2_7b import models as chinese_llama2_7b_model
    # from .models.hf_chinese_alpaca2_7b import models as chinese_alpaca2_7b_model
    # from .models.hf_belle2_13b import models as belle2_13b_model
    # from .models.hf_xverse_13b import models as xverse_13b_model
    # from .models.hf_ziya_llama_13b_v1 import models as ziya_llama_13b_v1_model
    # from .models.hf_llama2_13b_chinese_chat import models as llama2_13b_chinese_chat_model
    # from .models.hf_yulan_chat2_13b_fp16 import models as yulan_chat2_13b_fp16_model


    from .summarizers.mathbench import summarizer

    # from .eval_debug_aliyun import llmeval_infer as infer, llmeval_eval as eval
    # from .eval_debug_aliyun import llm_infer as infer, llm_eval as eval

datasets = sum([v for k, v in locals().items() if k.endswith("_datasets") or k == 'datasets'], [])
# datasets = [i for i in datasets if i.get('abbr', '').startswith('ceval-college_programming')]
# datasets = [i for i in datasets if not i.get('abbr', '').startswith('GaokaoBench')]

for d in datasets:
    d["infer_cfg"]["inferencer"]["save_every"] = 1

models = sum([v for k, v in locals().items() if k.endswith("_model")], [])

# for model in models:
#     model.update({"batch_padding":True, "batch_size":16})

# summarizer['dataset_abbrs'] = [i for i in summarizer['dataset_abbrs'] if isinstance(i, str) and not i.startswith('---------')]

from fnmatch import fnmatch

for m in models:
    # if fnmatch(m.get('abbr', ''), 'LLaMA-*B'):
    if 'llama-2' in m.get('abbr', '').lower():
        m['path'] = m['path'].replace('./models','/cpfs01/shared/public/public_hdd/llmeval/model_weights')
        m['tokenizer_path'] = m['tokenizer_path'].replace('./models', '/cpfs01/shared/public/public_hdd/llmeval/model_weights')
    elif m.get('abbr', '') not in ['gogpt-7b', 'alpaca-7b-hf']:
        m["tokenizer_kwargs"]["cache_dir"] = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub"
        m["model_kwargs"]["cache_dir"] = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub"
    #     # batch_padding
    #     # m["model_kwargs"]["batch_padding"] = True
    #     # trust_remote_code=True
        m["model_kwargs"]["trust_remote_code"] = True
        m["tokenizer_kwargs"]["trust_remote_code"] = True

        # local_files_only
        # m["tokenizer_kwargs"]["local_files_only"] = True

        # key = m['path'].split('/')[-1]
        # path = "/cpfs01/user/liuhongwei/work/LLMs/{key}".format(key=key)
        # m["tokenizer_kwargs"]["cache_dir"] = path
        # m["model_kwargs"]["cache_dir"] = path
        # m['path']=path
        # m['tokenizer_path']=path

# [{'type': HuggingFaceCausalLM, 'abbr': 'baichuan-7b-hf', 
# 'path': '/cpfs01/user/liuhongwei/LLMs/baichuan-7B', 
# 'tokenizer_path': '/cpfs01/user/liuhongwei/LLMs/baichuan-7B', 
# 'tokenizer_kwargs': {'padding_side': 'left', 'truncation_side': 'left', 
# 'trust_remote_code': True, 'use_fast': False, 'cache_dir': 
# '/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub'}, 
# 'max_out_len': 100, 'max_seq_len': 2048, 'batch_size': 8, 
# 'model_kwargs': {'device_map': 'auto', 'trust_remote_code': True, 
# 'cache_dir': '/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub', 
# 'batch_padding': True}, 'run_cfg': {'num_gpus': 1, 'num_procs': 1}}]

work_dir = './outputs/doyouknow/'
