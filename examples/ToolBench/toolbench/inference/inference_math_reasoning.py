"""
Inference script for math reasoning datasets
Generates predictions for SVAMP, GSM8K, NumGLUE, and Mathematica datasets
"""
import argparse
import json
import os
import sys
from tqdm import tqdm
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from toolbench.inference.LLM.tool_llama_model import ToolLLaMA
from toolbench.inference.LLM.tool_llama_lora_model import ToolLLaMALoRA
from toolbench.inference.LLM.llama_model import LlamaModel
from toolbench.model.model_adapter import get_conversation_template
from toolbench.tooleval.dataset.math_datasets import (
    load_svamp, load_gsm8k, load_numglue, load_mathematica,
    format_question_for_model, get_item_id
)


def load_model(args):
    """Load model based on arguments."""
    print(f"Loading model from: {args.model_path}")
    
    if args.lora:
        if not args.lora_path:
            raise ValueError("--lora_path must be specified when --lora is set")
        print(f"Loading LoRA adapter from: {args.lora_path}")
        model = ToolLLaMALoRA(
            model_name_or_path=args.model_path,
            lora_path=args.lora_path,
            template=args.template,
            device=args.device,
            max_sequence_length=args.max_sequence_length
        )
    elif args.use_toolllama:
        model = ToolLLaMA(
            model_name_or_path=args.model_path,
            template=args.template,
            device=args.device,
            max_sequence_length=args.max_sequence_length
        )
    else:
        model = LlamaModel(
            model_name_or_path=args.model_path,
            template=args.template,
            device=args.device,
            max_sequence_length=args.max_sequence_length
        )
    
    print("Model loaded successfully!")
    return model


def format_prompt(question: str, template: str = "tool-llama-single-round") -> str:
    """Format question into a prompt for the model."""
    conv = get_conversation_template(template)
    
    # For math reasoning, we use a simple user-assistant format
    # Check template structure
    if hasattr(conv, 'roles'):
        if len(conv.roles) >= 2:
            user_role = conv.roles[1] if len(conv.roles) > 1 else "User"
            assistant_role = conv.roles[-1] if len(conv.roles) > 1 else "Assistant"
        else:
            user_role = "User"
            assistant_role = "Assistant"
    else:
        user_role = "User"
        assistant_role = "Assistant"
    
    # Format as a simple Q&A prompt
    prompt = f"{user_role}: {question}\n{assistant_role}:"
    
    return prompt


def generate_prediction(model, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate prediction using the model."""
    try:
        # Use the model's prediction method
        if hasattr(model, 'prediction'):
            response = model.prediction(prompt, stop=None)
        elif hasattr(model, 'model') and hasattr(model, 'tokenizer'):
            # Fallback: try to use generate_stream directly
            from toolbench.inference.utils import generate_stream, SimpleChatIO
            device = "cuda" if (hasattr(model, 'use_gpu') and model.use_gpu) else "cpu"
            max_seq_len = getattr(model, 'max_sequence_length', 2048)
            
            gen_params = {
                "model": "",
                "prompt": prompt,
                "temperature": 0.7,
                "max_new_tokens": max_new_tokens,
                "stop": "</s>",
                "stop_token_ids": None,
                "echo": False
            }
            chatio = SimpleChatIO()
            output_stream = generate_stream(
                model.model, model.tokenizer, gen_params, 
                device, 
                max_seq_len, 
                force_generate=True
            )
            response = chatio.return_output(output_stream)
        else:
            raise ValueError("Model does not have prediction method or model/tokenizer attributes")
        
        return response.strip()
    except Exception as e:
        print(f"Error generating prediction: {e}")
        import traceback
        traceback.print_exc()
        return ""


def load_dataset(dataset_name: str, dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset based on name."""
    loaders = {
        'svamp': load_svamp,
        'gsm8k': load_gsm8k,
        'numglue': load_numglue,
        'mathematica': load_mathematica,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")
    
    print(f"Loading {dataset_name} dataset from: {dataset_path}")
    data = loaders[dataset_name](dataset_path)
    print(f"Loaded {len(data)} items")
    return data


def run_inference(
    model,
    dataset_name: str,
    dataset_path: str,
    output_path: str,
    template: str = "tool-llama-single-round",
    max_new_tokens: int = 512,
    batch_size: int = 1
):
    """Run inference on math reasoning dataset."""
    # Load dataset
    data = load_dataset(dataset_name, dataset_path)
    
    # Prepare output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Generate predictions
    predictions = {}
    
    print(f"\nGenerating predictions for {len(data)} items...")
    for i, item in enumerate(tqdm(data, desc="Generating predictions")):
        # Get question
        question = format_question_for_model(item, dataset_name)
        if not question:
            print(f"Warning: Empty question for item {i}, skipping")
            continue
        
        # Get item ID
        item_id = get_item_id(item, dataset_name, i)
        
        # Format prompt
        prompt = format_prompt(question, template)
        
        # Generate prediction
        try:
            response = generate_prediction(model, prompt, max_new_tokens)
            predictions[item_id] = response
        except Exception as e:
            print(f"Error processing item {item_id}: {e}")
            predictions[item_id] = ""
    
    # Save predictions
    print(f"\nSaving predictions to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(predictions)} predictions")
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for math reasoning datasets')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model directory')
    parser.add_argument('--lora', action='store_true',
                        help='Use LoRA model')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to LoRA adapter (required if --lora is set)')
    parser.add_argument('--use_toolllama', action='store_true',
                        help='Use ToolLLaMA class instead of LlamaModel')
    parser.add_argument('--template', type=str, default='tool-llama-single-round',
                        help='Conversation template to use')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to run inference on')
    parser.add_argument('--max_sequence_length', type=int, default=2048,
                        help='Maximum sequence length')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['svamp', 'gsm8k', 'numglue', 'mathematica'],
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset file')
    
    # Output arguments
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save predictions (JSON format)')
    
    # Generation arguments
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (currently not used, for future support)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.lora and not args.lora_path:
        parser.error("--lora_path must be specified when --lora is set")
    
    # Load model
    model = load_model(args)
    
    # Run inference
    run_inference(
        model=model,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        template=args.template,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()

