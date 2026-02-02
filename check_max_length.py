import json, transformers
model_path = "/mnt/lm_data_afs/wangzining/charles/models/Llama-2-7b-hf"
paths = [
    "examples/ToolBench/data/toolllama_G123_dfs_train.json",
    "examples/ToolBench/data/toolllama_G123_dfs_eval.json",
]
# ________________________
tok = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False, model_max_length=20000, padding_side="right")
max_len = -1
max_file = ""
for p in paths:
    with open(p, "r") as f:
        data = json.load(f)
    for ex in data:
        text = "".join(m["value"] for m in ex["conversations"])
        ids = tok(text, add_special_tokens=False, truncation=False).input_ids
        l = len(ids)
        if l > max_len:
            max_len, max_file = l, p
print(f"max tokens: {max_len} (from {max_file})")
