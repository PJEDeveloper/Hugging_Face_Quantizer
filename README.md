# Hugging Face Quantizer
Quantizer for Hugging Face Models

# ⚙️ Hugging Face 4-Bit Model Quantizer

This project enables **memory-efficient quantization of Hugging Face transformer models** to 4-bit using `BitsAndBytes`. It supports NF4 quantization with double precision, making large language models (LLMs) significantly smaller and more deployable on consumer GPUs.

---

## 🔍 Overview

The script and accompanying notebook simplify the process of:
- Loading a Hugging Face model (default: `Mistral-Nemo-Instruct-2407`)
- Applying 4-bit NF4 quantization with optional double quant
- Automatically managing GPU memory with `device_map="auto"`
- Saving both model and tokenizer to a user-specified path

---

## 🧠 Key Features

- 🧱 4-bit `nf4` quantization via `BitsAndBytesConfig`
- ⚙️ Auto device mapping (GPU/CPU as available)
- 🔁 Compatible with `transformers.AutoModelForCausalLM`
- 🖴 Saves quantized models locally
- 📄 Optional `.ipynb` interface for experimentation
- ✅ Ready-to-run CLI script

---

## 🗂 Project Structure

```
.
├── HF-4_bit_quantization.py         # Script version
├── HF_4_bit_quantization.ipynb      # Notebook version
├── requirements.txt                 # Pip dependencies
├── c_hf_q_env.yml                   # Conda environment
```

---

## 📦 Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using Conda (Recommended)

```bash
conda env create -f c_hf_q_env.yml
conda activate hf-quantizer
```

---

## 🚀 Run the Quantizer

```bash
python HF-4_bit_quantization.py
```

You will be prompted to:
- Provide a **directory path** for saving the model
- Specify a **folder name** to create within that path

Example output:
```
Model saved to: /home/user/quantized_models/mistral_4bit
```

---

## 🧪 Notebook Version

Open the Jupyter Notebook for inline testing:

```bash
jupyter notebook HF_4_bit_quantization.ipynb
```

---

## 🧰 Default Model Used

- [`mistralai/Mistral-Nemo-Instruct-2407`](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)

To modify:
```python
quantize_model_notebook(model_name="your/custom-model")
```

---

## 📝 License

This project is licensed under the Apache 2.0 License.

---

### Model Attribution & Licenses
This application incorporates the following pre-trained models:

Mistral-Nemo-Instruct-2407 License: Apache 2.0

Copyright (c) 2025, Patrick Hill

> Built by Patrick Hill — AI Developer | Model Optimization & LLM Deployment  
> [LinkedIn](https://www.linkedin.com/in/patrick-hill-4b9807178/)
