

# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# ======================================
# Notes by Shilong Liu:
# The scripts is adopted from the llava repo. original path: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py
# checkout the git history for the modification details.
# with two modifications:
#  1. support multiple image folders, separated by ','
#  2. support multiple dataset files, separated by ','
#  None that there might be multiple images with the same name in different folders, in this case, the first one will be used, which may not be the desired one!
# ======================================

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
# 在imports部分添加
import random
from sklearn.model_selection import train_test_split

import transformers
    # 保存分割后的数据到临时文件
import tempfile
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import reorganize_source_for_tool_use_batch, tokenizer_image_token

from PIL import Image


local_rank = None

# 在imports后添加疾病类别映射
DISEASE_CATEGORIES = [
    "cataract", "congenital_developmental", "dr", "glaucoma", 
    "inflammatory_retinal", "macular", "myopia", "optic_nerve_disorder", "retinal_vascular"
]
DISEASE_TO_ID = {disease: i for i, disease in enumerate(DISEASE_CATEGORIES)}
ID_TO_DISEASE = {i: disease for i, disease in enumerate(DISEASE_CATEGORIES)}

def extract_disease_label(text):
    """增强的疾病标签提取函数"""
    if not text:
        return -1
    
    text_lower = text.lower()
    
    # 首先尝试直接匹配标准类别
    for disease in DISEASE_CATEGORIES:
        if disease in text_lower:
            return DISEASE_TO_ID[disease]
    
    # 扩展的疾病名称映射
    disease_mapping = {
        "inflammatory_infectious_retinal": "inflammatory_retinal",
        "congenital_developmental_eye_diseases": "congenital_developmental", 
        "diabetic_retinopathy": "dr",
        "macular_diseases": "macular",
        "myopia_related": "myopia",
        "optic_nerve_disorders": "optic_nerve_disorder",
        "retinal_vascular_diseases": "retinal_vascular",
        "diabetic": "dr",
        "diabetes": "dr",
        "retinopathy": "dr",
        "myopic": "myopia",
        "nearsightedness": "myopia",
        "shortsightedness": "myopia",
        "age-related": "macular",
        "amd": "macular",
        "macula": "macular",
        "optic": "optic_nerve_disorder",
        "nerve": "optic_nerve_disorder",
        "papilledema": "optic_nerve_disorder",
        "disc": "optic_nerve_disorder",
        "retinal": "retinal_vascular",
        "vascular": "retinal_vascular",
        "vessel": "retinal_vascular",
        "hemorrhage": "retinal_vascular",
        "bleeding": "retinal_vascular",
        "inflammatory": "inflammatory_retinal",
        "infectious": "inflammatory_retinal",
        "inflammation": "inflammatory_retinal",
        "infection": "inflammatory_retinal",
        "congenital": "congenital_developmental",
        "developmental": "congenital_developmental",
        "birth": "congenital_developmental",
        "genetic": "congenital_developmental",
    }
    
    # 尝试映射表
    for variant, standard in disease_mapping.items():
        if variant in text_lower:
            return DISEASE_TO_ID[standard]
    
    return -1  # 未找到匹配的疾病

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

class ConcatDatasetPlus(ConcatDataset):

    @property
    def modality_lengths(self):
        length_list = []
        for d in self.datasets:
            length_list.extend(d.modality_lengths)

        return length_list



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# 修改make_supervised_data_module函数
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset

    # 处理数据路径
    data_path = data_args.data_path
    data_path_list = [i.strip() for i in data_path.split(',')]
    data_path_list = [x for x in data_path_list if x != ""]

    # 检查是否存在test_level1.json作为验证集
    validation_file = "./train_data_txt/test_level1.json"
    
    if os.path.exists(validation_file):
        print(f"Using specified validation file: {validation_file}")
        
        # 加载训练数据
        all_train_samples = []
        for data_name in data_path_list:
            assert os.path.exists(data_name), f"{data_name} does not exist"
            with open(data_name, 'r', encoding='utf-8') as f:
                data_samples = json.load(f)
                all_train_samples.extend(data_samples)
        
        # 加载指定的验证数据
        with open(validation_file, 'r', encoding='utf-8') as f:
            validation_samples = json.load(f)
        
        print(f"Total training samples: {len(all_train_samples)}")
        print(f"Total validation samples: {len(validation_samples)}")
        
        # 分析训练集疾病分布
        train_disease_dist = {}
        for sample in all_train_samples:
            disease_label = -1
            if len(sample.get("conversations", [])) >= 2:
                gpt_response = sample["conversations"][1]["value"]
                disease_label = extract_disease_label(gpt_response)
            
            if disease_label >= 0:
                disease_name = ID_TO_DISEASE[disease_label]
                train_disease_dist[disease_name] = train_disease_dist.get(disease_name, 0) + 1
            else:
                train_disease_dist["unknown"] = train_disease_dist.get("unknown", 0) + 1
        
        # 分析验证集疾病分布
        eval_disease_dist = {}
        for sample in validation_samples:
            disease_label = -1
            if len(sample.get("conversations", [])) >= 2:
                gpt_response = sample["conversations"][1]["value"]
                disease_label = extract_disease_label(gpt_response)
            
            if disease_label >= 0:
                disease_name = ID_TO_DISEASE[disease_label]
                eval_disease_dist[disease_name] = eval_disease_dist.get(disease_name, 0) + 1
            else:
                eval_disease_dist["unknown"] = eval_disease_dist.get("unknown", 0) + 1
        
        # 打印疾病分布
        print("\nTraining set disease distribution:")
        for disease, count in sorted(train_disease_dist.items()):
            print(f"  {disease}: {count} samples")
        
        print("\nValidation set disease distribution:")
        for disease, count in sorted(eval_disease_dist.items()):
            print(f"  {disease}: {count} samples")
        
        # 计算比例
        total_samples = len(all_train_samples) + len(validation_samples)
        train_ratio = len(all_train_samples) / total_samples
        eval_ratio = len(validation_samples) / total_samples
        print(f"\nDataset split ratio:")
        print(f"  Train: {len(all_train_samples)} samples ({train_ratio:.2%})")
        print(f"  Eval: {len(validation_samples)} samples ({eval_ratio:.2%})")
        
        # 为了保持比例一致，可以选择性地对训练集进行下采样
        # 如果需要保持特定比例，在这里添加下采样逻辑
        target_eval_ratio = 0.2  # 目标验证集比例
        
        if eval_ratio < target_eval_ratio:
            # 如果验证集比例过小，从训练集中移除一些样本以保持比例
            target_total = len(validation_samples) / target_eval_ratio
            target_train_size = int(target_total - len(validation_samples))
            
            if target_train_size < len(all_train_samples):
                print(f"\nAdjusting training set size to maintain ratio...")
                
                # 按疾病类别进行分层下采样
                train_disease_samples = {}
                train_no_disease = []
                
                for sample in all_train_samples:
                    disease_label = -1
                    if len(sample.get("conversations", [])) >= 2:
                        gpt_response = sample["conversations"][1]["value"]
                        disease_label = extract_disease_label(gpt_response)
                    
                    if disease_label >= 0:
                        disease_name = ID_TO_DISEASE[disease_label]
                        if disease_name not in train_disease_samples:
                            train_disease_samples[disease_name] = []
                        train_disease_samples[disease_name].append(sample)
                    else:
                        train_no_disease.append(sample)
                
                # 计算每个疾病类别的下采样数量
                adjusted_train_samples = []
                reduction_ratio = target_train_size / len(all_train_samples)
                
                for disease, samples in train_disease_samples.items():
                    keep_count = max(1, int(len(samples) * reduction_ratio))
                    random.shuffle(samples)
                    adjusted_train_samples.extend(samples[:keep_count])
                    print(f"  {disease}: {len(samples)} -> {keep_count} samples")
                
                if train_no_disease:
                    keep_count = max(1, int(len(train_no_disease) * reduction_ratio))
                    random.shuffle(train_no_disease)
                    adjusted_train_samples.extend(train_no_disease[:keep_count])
                    print(f"  unknown: {len(train_no_disease)} -> {keep_count} samples")
                
                all_train_samples = adjusted_train_samples
                print(f"\nAdjusted training set size: {len(all_train_samples)}")
                
                # 重新计算比例
                total_samples = len(all_train_samples) + len(validation_samples)
                train_ratio = len(all_train_samples) / total_samples
                eval_ratio = len(validation_samples) / total_samples
                print(f"Final split ratio:")
                print(f"  Train: {len(all_train_samples)} samples ({train_ratio:.2%})")
                print(f"  Eval: {len(validation_samples)} samples ({eval_ratio:.2%})")
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp(prefix="llava_fixed_split_")
        train_file = os.path.join(temp_dir, "train_data.json")
        eval_file = os.path.join(temp_dir, "eval_data.json")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(all_train_samples, f, ensure_ascii=False, indent=2)
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(validation_samples, f, ensure_ascii=False, indent=2)
        
        print(f"\nSplit data saved to:")
        print(f"  Train: {train_file}")
        print(f"  Eval: {eval_file}")
        
    else:
        print(f"Validation file {validation_file} not found.")
        print("Falling back to automatic stratified split...")
        
        # 保持原有的自动分层分割逻辑
        all_data_samples = []
        for data_name in data_path_list:
            assert os.path.exists(data_name), f"{data_name} does not exist"
            with open(data_name, 'r', encoding='utf-8') as f:
                data_samples = json.load(f)
                all_data_samples.extend(data_samples)
        
        print(f"Total samples loaded: {len(all_data_samples)}")
        
        # 按照疾病类别进行分层采样，确保训练集和验证集中都有各种疾病
        disease_samples = {}
        no_disease_samples = []
        
        # 按疾病分类分组
        for sample in all_data_samples:
            disease_label = -1
            if len(sample.get("conversations", [])) >= 2:
                gpt_response = sample["conversations"][1]["value"]
                disease_label = extract_disease_label(gpt_response)
            
            if disease_label >= 0:
                disease_name = ID_TO_DISEASE[disease_label]
                if disease_name not in disease_samples:
                    disease_samples[disease_name] = []
                disease_samples[disease_name].append(sample)
            else:
                no_disease_samples.append(sample)
        
        # 打印每种疾病的样本数量
        print("Disease distribution:")
        total_disease_samples = 0
        for disease, samples in disease_samples.items():
            print(f"  {disease}: {len(samples)} samples")
            total_disease_samples += len(samples)
        print(f"  unknown/no_disease: {len(no_disease_samples)} samples")
        print(f"  Total disease samples: {total_disease_samples}")
        
        # 分层采样：从每种疾病中分出20%作为验证集
        all_train_samples = []
        validation_samples = []
        
        for disease, samples in disease_samples.items():
            if len(samples) >= 5:  # 如果样本数>=5，进行分割
                # 确保每种疾病至少有1个验证样本
                n_eval = max(1, int(len(samples) * 0.2))
                n_train = len(samples) - n_eval
                
                # 随机分割
                random.shuffle(samples)
                disease_train = samples[:n_train]
                disease_eval = samples[n_train:n_train + n_eval]
                
                all_train_samples.extend(disease_train)
                validation_samples.extend(disease_eval)
                
                print(f"  {disease}: {len(disease_train)} train, {len(disease_eval)} eval")
            else:
                # 样本太少，全部放入训练集
                all_train_samples.extend(samples)
                print(f"  {disease}: {len(samples)} train, 0 eval (too few samples)")
        
        # 处理无法识别疾病的样本
        if no_disease_samples:
            if len(no_disease_samples) >= 5:
                n_eval = max(1, int(len(no_disease_samples) * 0.2))
                random.shuffle(no_disease_samples)
                all_train_samples.extend(no_disease_samples[:-n_eval])
                validation_samples.extend(no_disease_samples[-n_eval:])
                print(f"  unknown: {len(no_disease_samples) - n_eval} train, {n_eval} eval")
            else:
                all_train_samples.extend(no_disease_samples)
                print(f"  unknown: {len(no_disease_samples)} train, 0 eval (too few samples)")
        
        print(f"\nFinal split:")
        print(f"  Train samples: {len(all_train_samples)}")
        print(f"  Eval samples: {len(validation_samples)}")
        print(f"  Split ratio: {len(validation_samples)/(len(all_train_samples)+len(validation_samples)):.2%}")
        
        # 如果没有验证样本，从训练样本中随机选择一些作为验证集
        if len(validation_samples) == 0:
            print("Warning: No validation samples found. Creating validation set from training data...")
            n_eval = max(1, int(len(all_train_samples) * 0.2))
            random.shuffle(all_train_samples)
            validation_samples = all_train_samples[-n_eval:]
            all_train_samples = all_train_samples[:-n_eval]
            print(f"  Adjusted - Train: {len(all_train_samples)}, Eval: {len(validation_samples)}")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="llava_auto_split_")
        train_file = os.path.join(temp_dir, "train_split.json")
        eval_file = os.path.join(temp_dir, "eval_split.json")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(all_train_samples, f, ensure_ascii=False, indent=2)
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(validation_samples, f, ensure_ascii=False, indent=2)
        
        print(f"Split data saved to:")
        print(f"  Train: {train_file}")
        print(f"  Eval: {eval_file}")

    # 创建训练数据集
    train_data_args = copy.deepcopy(data_args)
    train_data_args.data_path = train_file
    train_dataset = build_dataset(train_data_args, tokenizer, dataset_cls)
    
    # 创建验证数据集
    eval_data_args = copy.deepcopy(data_args)
    eval_data_args.data_path = eval_file
    eval_dataset = build_dataset(eval_data_args, tokenizer, dataset_cls)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1 #modified
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) 
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) 
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    
    # reorganize sources by merging thoughts, actions, value into value, and add prefixs to value.
    sources = reorganize_source_for_tool_use_batch(sources)
    
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'images' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def load_image(self, image_file, image_folder):
        # load image from image_folder. support multiple folders
        # if multiple folders are provided, we will search the image in the order of the folders
        # the multiple folders should be separated by ','
        # **Warning**: if multiple folders has the same image name, the first one will be used, which may not be the desired one!
        if ',' not in image_folder:
            return Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        
        dir_list = image_folder.split(',')
        for dirname in dir_list:
            img_path = os.path.join(dirname.strip(), image_file)
            if os.path.exists(img_path):
                return Image.open(img_path).convert('RGB')

        raise ValueError("Unknow_file: {}".format(image_file))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # 提取疾病标签
        disease_label = -1
        if len(sources[0]["conversations"]) >= 2:
            gpt_response = sources[0]["conversations"][1]["value"]
            disease_label = extract_disease_label(gpt_response)

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = self.load_image(image_file, image_folder)
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # 添加疾病标签
        data_dict['disease_labels'] = torch.tensor(disease_label, dtype=torch.long)

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['images'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        # 收集疾病标签
        disease_labels = torch.stack([instance['disease_labels'] for instance in instances])
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            disease_labels=disease_labels,
        )

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def build_dataset(data_args, tokenizer, dataset_cls):
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    return train_dataset


# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
#                                 data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     dataset_cls = LazySupervisedDataset


#     #  concat data files
#     data_path = data_args.data_path
#     data_path_list = [i.strip() for i in data_path.split(',')]
#     data_path_list = [x for x in data_path_list if x != ""]

#     data_set_list = []
#     for data_name in data_path_list:
#         assert os.path.exists(data_name), f"{data_name} does not exist"
#         new_data_args = copy.deepcopy(data_args)
#         new_data_args.data_path = data_name
#         train_dataset_i = build_dataset(new_data_args, tokenizer, dataset_cls)
#         data_set_list.append(train_dataset_i)
#     train_dataset = ConcatDataset(data_set_list)
#     print(f"train_dataset size: {len(train_dataset)}")


import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch

class DiseaseClassificationTrainer(LLaVATrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 获取模型的hidden size
        if hasattr(self.model, 'config'):
            hidden_size = self.model.config.hidden_size
        else:
            hidden_size = 4096  # 默认值
            
        print(f"Initializing disease classifier with hidden_size: {hidden_size}")
        
        # 确定设备和数据类型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 获取模型的数据类型
        model_dtype = torch.float32  # 默认
        if hasattr(self.model, 'dtype'):
            model_dtype = self.model.dtype
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'torch_dtype'):
            model_dtype = self.model.config.torch_dtype
        else:
            # 尝试从模型参数中推断数据类型
            for param in self.model.parameters():
                model_dtype = param.dtype
                break
        
        print(f"Detected model dtype: {model_dtype}")
        
        # 设计更深层的分类头：4096 -> 1024 -> 512 -> 256 -> 9
        self.disease_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(256, len(DISEASE_CATEGORIES))
        )
        
        # 计算类别权重 - 基于训练数据分布
        class_counts = {
            0: 28,   # cataract
            1: 40,   # congenital_developmental  
            2: 101,  # dr
            3: 28,   # glaucoma
            4: 121,  # inflammatory_retinal
            5: 80,   # macular
            6: 80,   # myopia
            7: 120,  # optic_nerve_disorder
            8: 40    # retinal_vascular
        }
        
        # 计算逆频率权重
        total_samples = sum(class_counts.values())
        class_weights = []
        for i in range(len(DISEASE_CATEGORIES)):
            if i in class_counts:
                weight = total_samples / (len(DISEASE_CATEGORIES) * class_counts[i])
                class_weights.append(weight)
            else:
                class_weights.append(1.0)
        
        # 创建class_weights张量时直接使用正确的数据类型
        self.class_weights = torch.tensor(class_weights, dtype=model_dtype, device=device)
        print(f"Class weights: {self.class_weights}")
        print(f"Class weights dtype: {self.class_weights.dtype}")
        
        # 将分类头移到正确的设备和数据类型
        self.disease_classifier = self.disease_classifier.to(device=device, dtype=model_dtype)
        
        print(f"Disease classifier moved to device: {device}, dtype: {model_dtype}")
        
        # 确保分类头参与训练
        self.disease_classifier.requires_grad_(True)
        
        # 改进的初始化策略
        for module in self.disease_classifier.modules():
            if isinstance(module, torch.nn.Linear):
                # He初始化，适合ReLU激活函数
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.zeros_(module.bias)
        
        # 添加用于跟踪验证状态的变量
        self._evaluation_mode = False

    def save_disease_classifier(self, output_dir):
        """保存疾病分类头权重"""
        os.makedirs(output_dir, exist_ok=True)
        classifier_path = os.path.join(output_dir, "disease_classifier.bin")
        
        # 获取分类头的state_dict，处理DeepSpeed的情况
        classifier_state_dict = {}
        if self.disease_classifier is not None:
            for name, param in self.disease_classifier.named_parameters():
                # 使用maybe_zero_3处理DeepSpeed的参数
                classifier_state_dict[name] = maybe_zero_3(param, ignore_status=True)
        
        # 保存分类头权重
        torch.save(classifier_state_dict, classifier_path)
        print(f"Disease classifier saved to: {classifier_path}")
        
        # 同时保存疾病类别信息，便于测试时验证
        categories_path = os.path.join(output_dir, "disease_categories.json")
        disease_info = {
            "categories": DISEASE_CATEGORIES,
            "disease_to_id": DISEASE_TO_ID,
            "id_to_disease": ID_TO_DISEASE,
            "num_categories": len(DISEASE_CATEGORIES),
            "class_weights": self.class_weights.cpu().float().tolist()  # 转换为float保存
        }
        with open(categories_path, 'w', encoding='utf-8') as f:
            json.dump(disease_info, f, ensure_ascii=False, indent=2)
        print(f"Disease categories info saved to: {categories_path}")

    def _save_checkpoint(self, model, trial, metrics=None):
        """重写checkpoint保存，同时保存分类头"""
        # 调用原始的checkpoint保存
        checkpoint_folder = super()._save_checkpoint(model, trial, metrics)
        
        # 保存分类头到checkpoint文件夹
        if checkpoint_folder and (self.args.local_rank == 0 or self.args.local_rank == -1):
            self.save_disease_classifier(checkpoint_folder)
        
        return checkpoint_folder

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        组合语言建模损失和疾病分类损失
        """
        # 保存disease_labels
        disease_labels = inputs.get("disease_labels", None)

        # 移除disease_labels，避免传递给原始模型
        inputs_for_model = {k: v for k, v in inputs.items() if k != "disease_labels"}

        # 只进行一次前向传播，获取outputs和hidden_states
        outputs = model(**inputs_for_model, output_hidden_states=True, return_dict=True)

        # 计算语言建模损失 - 使用标准方法
        lm_loss = None
        if "labels" in inputs_for_model:
            labels = inputs_for_model["labels"]
            # 获取logits
            logits = outputs.logits

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens - 确保维度匹配
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

            # 重新调整形状，确保batch size匹配
            batch_size, seq_len_minus_1, vocab_size = shift_logits.shape
            shift_logits_flat = shift_logits.view(-1, vocab_size)
            shift_labels_flat = shift_labels.view(-1)

            # 确保两者的长度一致
            min_len = min(shift_logits_flat.size(0), shift_labels_flat.size(0))
            shift_logits_flat = shift_logits_flat[:min_len]
            shift_labels_flat = shift_labels_flat[:min_len]

            # 确保在同一设备上
            shift_labels_flat = shift_labels_flat.to(shift_logits_flat.device)

            # 计算损失
            lm_loss = loss_fct(shift_logits_flat, shift_labels_flat)

        total_loss = lm_loss if lm_loss is not None else torch.tensor(0.0, device=outputs.logits.device, requires_grad=True)

        # 添加疾病分类损失
        if disease_labels is not None and (disease_labels >= 0).any():
            try:
                # 使用已有的hidden states（避免重复前向传播）
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    last_hidden_states = outputs.hidden_states[-1]
                    attention_mask = inputs_for_model.get('attention_mask', None)

                    if attention_mask is not None:
                        sequence_lengths = attention_mask.sum(dim=1) - 1
                        batch_size = last_hidden_states.size(0)
                        max_len = last_hidden_states.size(1) - 1
                        sequence_lengths = torch.clamp(sequence_lengths, 0, max_len)
                        pooled_features = last_hidden_states[range(batch_size), sequence_lengths]
                    else:
                        pooled_features = last_hidden_states.mean(dim=1)

                    # 确保分类头在正确的设备和数据类型上
                    self.disease_classifier = self.disease_classifier.to(
                        device=pooled_features.device, 
                        dtype=pooled_features.dtype
                    )
                    
                    # 确保权重也在正确设备和数据类型上
                    self.class_weights = self.class_weights.to(
                        device=pooled_features.device,
                        dtype=pooled_features.dtype  # 关键修复：确保数据类型一致
                    )

                    disease_logits = self.disease_classifier(pooled_features)

                    valid_mask = disease_labels >= 0
                    if valid_mask.any():
                        # 确保disease_labels也是正确的数据类型
                        valid_disease_labels = disease_labels[valid_mask].to(
                            device=disease_logits.device
                        )
                        
                        classification_loss = F.cross_entropy(
                            disease_logits[valid_mask], 
                            valid_disease_labels,
                            weight=self.class_weights
                        )

                        # 增加分类损失权重，从 0.1 增加到 0.5
                        total_loss = total_loss + 0.5 * classification_loss

                        # 打印损失信息
                        if hasattr(self, '_step_count'):
                            self._step_count += 1
                        else:
                            self._step_count = 1

                        if self._step_count % 100 == 0:
                            print(f"Step {self._step_count}: LM Loss: {lm_loss:.4f}, Classification Loss: {classification_loss:.4f}")

            except Exception as e:
                print(f"Classification loss calculation failed: {e}")
                import traceback
                traceback.print_exc()

        # 恢复原始inputs
        inputs["disease_labels"] = disease_labels

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss
    
    def _prediction_step_disease_classification(self, model, inputs):
        """独立的疾病分类预测步骤，确保状态一致性"""
        # 确保模型处于评估模式
        was_training = model.training
        model.eval()
        
        # 确保分类头也处于评估模式
        self.disease_classifier.eval()
        
        try:
            with torch.no_grad():
                disease_labels = inputs.get("disease_labels", None)
                if disease_labels is None:
                    return None, None
                
                # 检查有效标签
                valid_mask = disease_labels >= 0
                if not valid_mask.any():
                    return None, None
                
                # 创建不包含disease_labels的输入
                model_inputs = {k: v for k, v in inputs.items() if k != "disease_labels"}
                
                # 确保输入tensor的数据类型和设备正确
                for key, value in model_inputs.items():
                    if isinstance(value, torch.Tensor):
                        model_inputs[key] = value.to(device=model.device)
                
                # 运行模型
                outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
                
                # 从hidden states中提取特征
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    last_hidden_states = outputs.hidden_states[-1]
                    attention_mask = model_inputs.get('attention_mask', None)
                    
                    if attention_mask is not None:
                        # 与训练时完全相同的特征提取方法
                        sequence_lengths = attention_mask.sum(dim=1) - 1
                        batch_size = last_hidden_states.size(0)
                        max_len = last_hidden_states.size(1) - 1
                        sequence_lengths = torch.clamp(sequence_lengths, 0, max_len)
                        pooled_features = last_hidden_states[range(batch_size), sequence_lengths]
                    else:
                        pooled_features = last_hidden_states.mean(dim=1)
                    
                    # 确保分类头在正确的设备和数据类型上
                    self.disease_classifier = self.disease_classifier.to(
                        device=pooled_features.device,
                        dtype=pooled_features.dtype
                    )
                    
                    # 通过分类头预测
                    disease_logits = self.disease_classifier(pooled_features)
                    predicted_diseases = torch.argmax(disease_logits, dim=1)
                    
                    # 确保所有tensor都在同一设备上
                    valid_mask = valid_mask.to(predicted_diseases.device)
                    disease_labels = disease_labels.to(predicted_diseases.device)
                    
                    # 返回有效的预测和真实标签
                    if valid_mask.any():
                        valid_predictions = predicted_diseases[valid_mask].cpu().numpy()
                        valid_true_labels = disease_labels[valid_mask].cpu().numpy()
                        return valid_predictions, valid_true_labels
                    
                return None, None
                
        except Exception as e:
            print(f"Error in disease classification prediction: {e}")
            return None, None
        finally:
            # 恢复原始训练状态
            if was_training:
                model.train()
                self.disease_classifier.train()
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        修复的评估函数，确保状态一致性
        """
        print("Starting FIXED evaluation with disease classification...")
        
        # 设置评估模式标志
        self._evaluation_mode = True
        
        # 确保模型处于评估模式
        self.model.eval()
        self.disease_classifier.eval()
        
        # 调用原始评估
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 使用 self.eval_dataset 如果 eval_dataset 参数为 None
        dataset_to_use = eval_dataset if eval_dataset is not None else getattr(self, 'eval_dataset', None)
        
        # 计算疾病分类准确率
        if dataset_to_use is not None:
            print("Computing disease classification accuracy with FIXED evaluation...")
            predictions = []
            true_labels = []
            total_samples = 0
            valid_samples = 0
            
            eval_dataloader = self.get_eval_dataloader(dataset_to_use)
            
            # 再次确保模型处于评估模式
            self.model.eval()
            self.disease_classifier.eval()
            
            for batch_idx, inputs in enumerate(eval_dataloader):
                try:
                    inputs = self._prepare_inputs(inputs)
                    
                    # 使用独立的预测步骤
                    batch_predictions, batch_true_labels = self._prediction_step_disease_classification(
                        self.model, inputs
                    )
                    
                    if batch_predictions is not None and batch_true_labels is not None:
                        predictions.extend(batch_predictions)
                        true_labels.extend(batch_true_labels)
                        valid_samples += len(batch_predictions)
                    
                    total_samples += inputs.get("disease_labels", torch.tensor([])).numel()
                    
                    # 减少打印频率，避免干扰训练
                    if (batch_idx + 1) % 50 == 0:
                        current_acc = accuracy_score(true_labels, predictions) if len(predictions) > 0 else 0.0
                        print(f"Evaluation progress: {batch_idx + 1} batches, current accuracy: {current_acc:.4f}")
                        
                except Exception as e:
                    if batch_idx < 3:  # 只打印前几个错误
                        print(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
            
            print(f"Total samples: {total_samples}")
            print(f"Valid samples: {valid_samples}")
            print(f"Predictions collected: {len(predictions)}")
            print(f"True labels collected: {len(true_labels)}")
            
            # 计算最终准确率
            if len(predictions) > 0:
                accuracy = accuracy_score(true_labels, predictions)
                eval_result[f"{metric_key_prefix}_disease_accuracy"] = accuracy
                print(f"FIXED Disease Classification Accuracy: {accuracy:.4f}")
                
                # 计算各类别准确率
                from collections import Counter
                pred_counter = Counter(predictions)
                true_counter = Counter(true_labels)
                
                print(f"Prediction distribution: {dict(pred_counter)}")
                print(f"True label distribution: {dict(true_counter)}")
                
            else:
                print("No valid predictions for disease classification")
                eval_result[f"{metric_key_prefix}_disease_accuracy"] = 0.0
        else:
            print("No eval dataset available for disease classification")
        
        # 清除评估模式标志
        self._evaluation_mode = False
        
        # 如果不是在最终评估，恢复训练模式
        if self.model.training:
            self.model.train()
            self.disease_classifier.train()
        
        return eval_result

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    # 在模型加载后添加
    model.config.output_hidden_states = True

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = DiseaseClassificationTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)  # 改为True
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

    # 训练结束后保存分类头权重
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        trainer.save_disease_classifier(training_args.output_dir)
        print("Final disease classifier weights saved!")

if __name__ == "__main__":
    train()