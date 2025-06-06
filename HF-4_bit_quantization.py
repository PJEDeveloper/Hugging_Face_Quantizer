# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path

# %%
def select_output_directory_terminal():
    print("Choose where to save the quantized model.")
    selected_dir = input("Enter the full path to the directory: ").strip()
    if not selected_dir:
        raise Exception("No directory selected.")
    
    folder_name = input("Enter a folder name for the model: ").strip()
    if not folder_name:
        raise Exception("No folder name provided.")

    return Path(selected_dir) / folder_name

# %%
def quantize_model_notebook(model_name="mistralai/Mistral-Nemo-Instruct-2407"):
    save_path = select_output_directory_terminal()
    save_path.mkdir(parents=True, exist_ok=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading model {model_name} with 4-bit quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")

# %%
if __name__ == "__main__":
    quantize_model_notebook()
