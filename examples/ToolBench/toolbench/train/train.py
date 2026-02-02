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

from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

import sys
# Add project root to sys.path to allow importing coat
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from coat.models.coat_llama_fake import LlamaForCausalLMFake

from toolbench.tool_conversation import SeparatorStyle
from toolbench.model.model_adapter import get_conversation_template
from toolbench.train.llama_condense_monkey_patch import replace_llama_with_condense

# Import inference and evaluation functions (will be checked later when needed)
INFERENCE_AVAILABLE = False
try:
    from toolbench.inference.inference_math_reasoning import run_inference, load_model as load_inference_model
    from toolbench.tooleval.eval_math_reasoning import evaluate_dataset
    INFERENCE_AVAILABLE = True
except ImportError:
    # Will print warning later when actually used
    pass


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
torch.set_printoptions(profile="full")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_mxfp4_fake: bool = field(
        default=False,
        metadata={"help": "Use MXFP4 fake-quantized Llama model from coat."},
    )
    fabit: str = field(
        default="mxfp4_e2m1",
        metadata={"help": "Forward activation/weight fake quantization format (e.g., mxfp4_e2m1 or bf16)."},
    )
    babit: str = field(
        default="mxfp4_e2m1",
        metadata={"help": "Backward fake quantization format for activations/weights."},
    )
    backward_quantize: bool = field(
        default=False,
        metadata={"help": "Enable backward fake quantization (STE if False)."},
    )
    minus_exp: Optional[str] = field(
        default=None,
        metadata={"help": "Exponent offset for MXFP4 fake quantization. Accepts int, 'None', 'auto', or 'auto-reverse'."},
    )
    auto_reverse: bool = field(
        default=False,
        metadata={"help": "Enable auto reverse exponent heuristic for MX formats if supported."},
    )
    attn_quantize: bool = field(
        default=False,
        metadata={"help": "Apply additional fake quantization to attention QKV projections."},
    )
    attn_quantize_forward_bit: Optional[str] = field(
        default=None,
        metadata={"help": "Forward format for attention QKV fake quantization (defaults to fabit)."},
    )
    attn_quantize_backward_bit: Optional[str] = field(
        default=None,
        metadata={"help": "Backward format for attention QKV fake quantization (defaults to babit)."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    conv_template: str = field(
        default=None, metadata={"help": "Template used to format the training data."}
    )
    lazy_preprocess: bool = False
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # Allow optimizer-state quant args to be parsed (passed through CLI but unused here)
    first_order_expansion: str = field(default="false")
    second_order_expansion: str = field(default="false")
    first_order_bit: str = field(default="BF16")
    second_order_bit: str = field(default="BF16")
    source_model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Original maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Expanded maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    template: str="tool-llama"
) -> Dict:
    conv = get_conversation_template(template)
    if template == "tool-llama":
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
        roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())


    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0  # Llama 的 <unk> token id 通常是 0
    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    
    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[-1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                continue
            turn_len = len(tokenizer(turn, add_special_tokens=False).input_ids)

            parts = turn.split(sep)
            
            # only train on the last assistant reply, treat the history chat as instruction
            prefix = parts[:-1]
            instruction = ""
            for part in prefix:
                instruction += part
                instruction += sep

            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(instruction, add_special_tokens=False).input_ids)

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, template="tool-llama"):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        self.template = template
        data_dict = preprocess(sources, tokenizer, self.template)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, template="tool-llama"):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.template = template

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.template)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))
    if data_args.eval_data_path is not None:
        train_raw_data = raw_data
        eval_raw_data = json.load(open(data_args.eval_data_path, "r"))
    else:
        # Split train/test
        perm = np.random.permutation(len(raw_data))
        split = int(len(perm) * 0.98)
        train_indices = perm[:split]
        eval_indices = perm[split:]
        train_raw_data = [raw_data[i] for i in train_indices]
        eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def _parse_minus_exp(value):
    """Normalize minus_exp from CLI: accept ints, None, 'auto', or 'auto-reverse'."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"none", ""}:
            return None
        if stripped == "auto":
            return "auto"
        if stripped in {"auto-reverse", "auto_reverse"}:
            return "auto-reverse"
        return int(stripped)
    return int(value)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    parsed_minus_exp = _parse_minus_exp(model_args.minus_exp)
    model_args.minus_exp = parsed_minus_exp
    if training_args.source_model_max_length < training_args.model_max_length:
        condense_ratio = int(training_args.model_max_length/training_args.source_model_max_length)
        # ratio = N means the sequence length is expanded by N, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
        replace_llama_with_condense(ratio=condense_ratio)
    local_rank = training_args.local_rank

    # Load config; only switch to fake-quant path when fake args are provided
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    use_fake = any(
        x is not None
        for x in (
            model_args.fabit,
            model_args.babit,
            model_args.attn_quantize,
            model_args.minus_exp,
        )
    )

    if use_fake:
        # Rebuild as CoatLlamaFakeConfig so checkpoints carry the correct model_type
        config = CoatLlamaFakeConfig.from_dict(config.to_dict())
        config.model_type = CoatLlamaFakeConfig.model_type
        config.architectures = ["CoatLlamaFakeForCausalLM"]

        coat_fp8_args = getattr(config, "coat_fp8_args", {}) or {}
        if model_args.fabit is not None:
            coat_fp8_args["fabit"] = model_args.fabit
        if model_args.babit is not None:
            coat_fp8_args["babit"] = model_args.babit
        if model_args.attn_quantize is not None:
            coat_fp8_args["attn_quantize"] = model_args.attn_quantize
        if model_args.minus_exp is not None:
            coat_fp8_args["minus_exp"] = model_args.minus_exp
        config.coat_fp8_args = coat_fp8_args

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    if model_args.use_mxfp4_fake:
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        # Thread fake-quantization settings into the config so LlamaForCausalLMFake picks them up.
        config.fabit = model_args.fabit
        config.babit = model_args.babit
        config.backward_quantize = model_args.backward_quantize
        config.minus_exp = model_args.minus_exp
        config.auto_reverse = model_args.auto_reverse
        config.attn_quantize = model_args.attn_quantize
        config.attn_quantize_forward_bit = model_args.attn_quantize_forward_bit or model_args.fabit
        config.attn_quantize_backward_bit = model_args.attn_quantize_backward_bit or model_args.babit
        model = LlamaForCausalLMFake.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
            config=config,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map=device_map
        )
    # Debug: print key fake-quantization config to verify propagation (rank0 only).
    rank0_print(
        "[MXFP4 fake debug] config.fabit=",
        getattr(model.config, "fabit", None),
        "babit=",
        getattr(model.config, "babit", None),
        "backward_quantize=",
        getattr(model.config, "backward_quantize", None),
        "minus_exp=",
        getattr(model.config, "minus_exp", None),
        "auto_reverse=",
        getattr(model.config, "auto_reverse", None),
        "attn_quantize=",
        getattr(model.config, "attn_quantize", None),
        "attn_q_forward=",
        getattr(model.config, "attn_quantize_forward_bit", None),
        "attn_q_backward=",
        getattr(model.config, "attn_quantize_backward_bit", None),
    )
    model.config.use_cache = False
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
