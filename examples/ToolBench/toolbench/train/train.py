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
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            
            # only train on the last assistant reply, treat the history chat as instruction
            prefix = parts[:-1]
            instruction = ""
            for part in prefix:
                instruction += part
                instruction += sep

            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(instruction).input_ids) - 2

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


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.source_model_max_length < training_args.model_max_length:
        condense_ratio = int(training_args.model_max_length/training_args.source_model_max_length)
        # ratio = N means the sequence length is expanded by N, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
        replace_llama_with_condense(ratio=condense_ratio)
    local_rank = training_args.local_rank
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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map
    )
    model.config.use_cache = False
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    
    
    # Run inference and evaluation on math reasoning datasets (only on rank 0)
    if local_rank == 0 and INFERENCE_AVAILABLE:
        # Check if wandb is being used
        use_wandb = training_args.report_to and "wandb" in training_args.report_to
        
        # Get paths from environment variables or use defaults
        data_base_dir = os.environ.get("DATA_BASE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "math_datasets"))
        predictions_base_dir = os.environ.get("PREDICTIONS_BASE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "predictions", "math_reasoning"))
        os.makedirs(predictions_base_dir, exist_ok=True)
        
        # Dataset paths
        dataset_paths = {
            "svamp": os.path.join(data_base_dir, "svamp", "test.json"),
            "gsm8k": os.path.join(data_base_dir, "gsm8k", "test.jsonl"),
            "numglue": os.path.join(data_base_dir, "numglue", "test.json"),
            "mathematica": os.path.join(data_base_dir, "mathematica", "test.json"),
        }
        
        rank0_print("\n" + "="*50)
        rank0_print("Starting Math Reasoning Pipeline Test")
        rank0_print("="*50)
        
        # Run inference on all datasets
        rank0_print("\nRunning inference on math reasoning datasets...")
        for dataset_name, dataset_path in dataset_paths.items():
            if not os.path.exists(dataset_path):
                rank0_print(f"Warning: Dataset file not found: {dataset_path}, skipping {dataset_name}")
                continue
            
            output_path = os.path.join(predictions_base_dir, f"{dataset_name}_predictions.json")
            
            # Skip if predictions already exist and skip_existing is set
            skip_existing = os.environ.get("SKIP_EXISTING", "false").lower() == "true"
            if skip_existing and os.path.exists(output_path):
                rank0_print(f"Skipping {dataset_name} (predictions already exist)")
                continue
            
            try:
                rank0_print(f"\nRunning inference on {dataset_name}...")
                # Create a simple args object for inference
                class InferenceArgs:
                    def __init__(self):
                        self.model_path = training_args.output_dir
                        self.lora = False
                        self.lora_path = None
                        self.use_toolllama = True
                        self.template = data_args.conv_template or "tool-llama-single-round"
                        self.device = "cuda" if torch.cuda.is_available() else "cpu"
                        self.max_sequence_length = training_args.model_max_length
                        self.dataset = dataset_name
                        self.dataset_path = dataset_path
                        self.output_path = output_path
                        self.max_new_tokens = 512
                        self.batch_size = 1
                
                inference_args = InferenceArgs()
                # Create an adapter wrapper for trainer.model to work with inference functions
                # trainer.model is a transformers model, but inference expects ToolLLaMA-like interface
                class ModelAdapter:
                    def __init__(self, model, tokenizer, device, max_sequence_length):
                        self.model = model
                        self.tokenizer = tokenizer
                        self.device = device
                        self.max_sequence_length = max_sequence_length
                        self.use_gpu = (device == "cuda")
                
                # Unwrap model if it's wrapped by DDP/FSDP
                unwrapped_model = trainer.model
                if hasattr(unwrapped_model, 'module'):  # DDP wrapper
                    unwrapped_model = unwrapped_model.module
                
                # Set model to eval mode
                unwrapped_model.eval()
                
                inference_model = ModelAdapter(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    device=inference_args.device,
                    max_sequence_length=inference_args.max_sequence_length
                )
                run_inference(
                    model=inference_model,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    output_path=output_path,
                    template=inference_args.template,
                    max_new_tokens=inference_args.max_new_tokens,
                    batch_size=inference_args.batch_size
                )
                rank0_print(f"✓ Inference completed for {dataset_name}")
            except Exception as e:
                rank0_print(f"Error running inference on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Run evaluation on all datasets
        rank0_print("\nRunning evaluation on math reasoning datasets...")
        eval_results = {}
        for dataset_name, dataset_path in dataset_paths.items():
            predictions_path = os.path.join(predictions_base_dir, f"{dataset_name}_predictions.json")
            
            if not os.path.exists(predictions_path):
                rank0_print(f"Warning: Predictions file not found: {predictions_path}, skipping evaluation for {dataset_name}")
                continue
            
            if not os.path.exists(dataset_path):
                rank0_print(f"Warning: Dataset file not found: {dataset_path}, skipping evaluation for {dataset_name}")
                continue
            
            try:
                rank0_print(f"\nEvaluating {dataset_name}...")
                output_path = os.path.join(predictions_base_dir, "eval_results", dataset_name)
                accuracy, results = evaluate_dataset(
                    predictions_path=predictions_path,
                    dataset=dataset_name,
                    dataset_path=dataset_path,
                    output_path=output_path,
                    log_to_wandb=use_wandb
                )
                eval_results[dataset_name] = {
                    "accuracy": accuracy,
                    "correct": sum(1 for r in results if r['correct']),
                    "total": len(results)
                }
                rank0_print(f"✓ Evaluation completed for {dataset_name}: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
            except Exception as e:
                rank0_print(f"Error evaluating {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print summary
        if eval_results:
            rank0_print("\n" + "="*50)
            rank0_print("Math Reasoning Evaluation Summary")
            rank0_print("="*50)
            for dataset_name, results in eval_results.items():
                rank0_print(f"{dataset_name}: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%) - {results['correct']}/{results['total']}")
            
            # Log summary to wandb if available
            if use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        summary_metrics = {f"eval/{name}/accuracy": res["accuracy"] for name, res in eval_results.items()}
                        wandb.log(summary_metrics)
                        rank0_print("\n✓ Evaluation results logged to wandb")
                except Exception as e:
                    rank0_print(f"Warning: Failed to log summary to wandb: {e}")
        
        rank0_print("\n" + "="*50)
        rank0_print("Math Reasoning Pipeline Test Completed")
        rank0_print("="*50 + "\n")
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
