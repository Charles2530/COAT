import os

# File path to be fixed
target_file = "/mnt/lm_data_afs/wangzining/charles/COAT/coat/models/coat_llama_fake.py"

def fix_indentation_error():
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found.")
        return

    print(f"Opening file: {target_file}")
    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Define the content to insert
    patch_line = "    _update_causal_mask = LlamaModel._update_causal_mask\n"
    
    # Check if the patch is already applied to avoid duplicates
    if any(patch_line.strip() in line for line in lines):
        print("Warning: Patch already detected in the file. Skipping...")
        return

    # Find the insertion point: Before the next class definition
    insertion_idx = -1
    for i, line in enumerate(lines):
        if "class CoatLlamaFakeForCausalLM" in line:
            insertion_idx = i
            break

    if insertion_idx != -1:
        # Construct new content
        # Adding empty lines for better readability
        new_content = lines[:insertion_idx] + ["\n", patch_line, "\n"] + lines[insertion_idx:]
        
        with open(target_file, 'w', encoding='utf-8') as f:
            f.writelines(new_content)
        print("Success: Applied _update_causal_mask patch with correct indentation.")
    else:
        print("Error: Could not locate 'class CoatLlamaFakeForCausalLM' to perform insertion.")

if __name__ == "__main__":
    fix_indentation_error()
