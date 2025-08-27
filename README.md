# vLLM Mistral-7B Demo
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
- Python copies its interpreter and standard libraries into a new isolated folder (vllm-env/)
- It updates the shell’s $PATH so that when you type python or pip, it points to the copies inside vllm-env/, not your system ones.
- Because the environment is active, you’re upgrading pip inside the venv only.
This ensures you have the latest installer before you start adding packages like vllm or torch


### 2. Install PyTorch
<pre>
  pip install "vllm[torch]"
</pre>
- pip checks for the latest release of vllm and installs PyTorch
- Everything gets installed into the virtual environment vllm-env

### 3. Verify Setup
Switch to the python shell.
<pre>
  python
</pre>
- This opens the interactive Python REPL (Read-Eval-Print Loop). From here, you can type Python code line by line.

<br>

Type these commands. 
<pre>
  import torch
  print(torch.__version__)
  print(torch.cuda.is_available())
  print(torch.cude.get_device_name(0))
</pre>
System output:

<pre>
  2.7.1+cu126
  True
  NVIDIA A10
</pre>
- This should be the expected output. If the output doesn't match, there was an error during setup. 

### 4. Start the vLLM OpenAI compatible server

<pre>
  python3 -m vllm.entrypoints.openai.api_server
--model ./Mistral-7B-Instruct-v0.2-GPTQ \
--quantization gptq \
--dtype float16 \
--port 8000
</pre>
- This launches vLLM’s OpenAI-compatible server on port 8000, serving the locally stored Mistral-7B-Instruct v0.2 GPTQ model in fp16 precision.

<br>

## Errors Encountered
### Wrong dtype with GPTQ

First error I encountered was setting the dtype to "auto"
<pre>
  python3 -m vllm.entrypoints.openai.api_server
--model ./Mistral-7B-Instruct-v0.2-GPTQ \
--dtype auto \
--quantization gptq \
--port 8000
</pre>
"dtype auto" made vLLM choose bfloat16, since the GPU supports it. But GPTQ quantized models are not compatible with bfloat16.
- Deeper Explaination: GPTQ is a quantization method that pre-compresses weights (e.g., 4-bit or 8-bit). These quantized weights aren’t compatible with bfloat16 (bf16). Instead, GPTQ models are meant to be loaded in float16 (fp16) or int4/int8, depending on how they were built.

### SafeTensor Error: header too large


