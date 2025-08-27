# vLLM Mistral-7B Demo
Serving Mistral-7B with vLLM in a Python Virtual Environment

## PROJECT DESCRIPTION

This project documents how to set up and serve the **Mistral-7B-Instruct** model using the [vLLM](https://github.com/vllm-project/vllm) inference engine inside a Python virtual environment. The environment was tested on an **NVIDIA A10 GPU** cloud instance.

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

## Errors encountered
### Error #1: Wrong dtype with GPTQ

First attepmt at starting the vLLM server, I used this command.
<pre>
  python3 -m vllm.entrypoints.openai.api_server
--model ./Mistral-7B-Instruct-v0.2-GPTQ \
--dtype auto \
--quantization gptq \
--port 8000
</pre>
I encountered an error here: 
<pre>
  Value error, torch.bfloat16 is not supported for quantization method gpta. Supported dtypes: [torch.float16] [type=value_error, input_value=ArgsKwargs(O, {'model_co...additional_conf ig': {}}), input_type=ArgsKwargs]
</pre>
"dtype auto" made vLLM choose bfloat16, since the GPU supports it. But GPTQ quantized models are not compatible with bfloat16.
- Deeper Explaination: GPTQ is a quantization method that pre-compresses weights (e.g., 4-bit or 8-bit). These quantized weights aren’t compatible with bfloat16 (bf16). Instead, GPTQ models are meant to be loaded in float16 (fp16) or int4/int8, depending on how they were built.

### Error 2: SafeTensor error "header too large"

<pre>
  safetensors_rust.SafetensorError: Error while deserializing header: header too large
</pre>
- Explaination: 
- This error was caused because safetensors tried to read the file’s header (the metadata about tensors in the file), but it was corrupted or incomplete.
- “Header too large” usually happens when the model weights didn’t download correctly — often because of Git LFS (Large File Storage... more on that below).

To resolve this error, you can start by removing Mistral-7B and reinstalling it. This ensures that you are removing any files that might be accidentally corrupted. 
<pre>
  rm -rf Mistral-7B-Instruct-v0.2-GPTQ
  git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
</pre>

Before moving on to the next step and entering the Mistral file directory, you will want to ensure that Git LFS is installed in the virtual environment:
<pre>
  sudo apt install git-lfs
  git lfs install
</pre>

After Git LFS is installed, move into the Mistral file directory and run "Git LFS pull"
<pre>
  cd Mistral-7B-Instruct-v0.2-GPTQ
  git LFS pull
</pre>
- Hugging Face and GitHub use Git LFS for big files (like multi-GB model weights). If you clone a repo with big files without Git LFS, you only get pointer stubs (tiny text files, not the full model).
- When vLLM tries to load those stubs as real model weights → safetensors chokes → error
- Doing this process, Git will actually fetch the multi-GB .safetensors files instead of stubs and you should be able to proceed without error. 
