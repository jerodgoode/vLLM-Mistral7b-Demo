# vLLM-Mistral-Demo
Serving Mistral-7B with vLLM in a Python Virtual Environment

## PROJECT DESCRIPTION

This project documents how I set up and served the **Mistral-7B-Instruct** model using the [vLLM](https://github.com/vllm-project/vllm) inference engine inside a Python virtual environment. The environment was tested on an **NVIDIA A10 GPU** cloud instance.


## SETUP PROCESS

### 1. Create Python Virtual Environment
<pre>
python3 -m venv vllm-env 
source vllm-env/bin/activate 
python -m pip install --upgrade pip
</pre>
-Python copies its interpreter and standard libraries into a new isolated folder (vllm-env/)
<br> -It updates the shell’s $PATH so that when you type python or pip, it points to the copies inside vllm-env/, not your system ones.
<br> -Because the environment is active, you’re upgrading pip inside the venv only
This ensures you have the latest installer before you start adding packages like vllm or torch

### 2. Install PyTorch
