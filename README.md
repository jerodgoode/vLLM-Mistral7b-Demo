# vLLM Mistral-7B Demo
Serving Mistral-7B with vLLM in a Python virtual environment on NVIDIA A10

## PROJECT DESCRIPTION
This project documents how to set up and serve the **Mistral-7B-Instruct** model using the [vLLM](https://github.com/vllm-project/vllm) inference engine inside a Python virtual environment. The environment was tested on an **NVIDIA A10 GPU** cloud instance. Mistral-7B was chosen for this project due to its performance comparatively to the LLaMA 2 family, another LLM developed by Meta. Below is a comparison of Mistral-7B’s performance vs. LLaMA:

![source: https://mistral.ai/news/announcing-mistral-7b](https://raw.githubusercontent.com/jerodgoode/vLLM-Mistral7b-Demo/photos/Mistral7B.png)


## SETUP PROCESS

### 1. Create Python Virtual Environment
<pre>
  python3 -m venv vllm-env 
  source vllm-env/bin/activate 
  python -m pip install --upgrade pip
</pre>
- Python copies its interpreter and standard libraries into a new isolated folder (vllm-env/)
- It updates the shell’s $PATH so that when you type python or pip, it points to the copies inside vllm-env/, not your system ones.
- Upgrades pip inside the venv to ensure the latest installer before adding packages.


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

### 5. Querying the LLM
Now that Mistral-7B is up and running, test it using "curl"
<pre>
  curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./Mistral-7B-Instruct-v0.2-GPTQ",
    "messages":[{"role":"user","content":"Say hello in 5 words."}]
  }'
</pre>

Mistral-7B will send back a text output response. This was the response received: 
<pre>
  {
    "id":"chatcmpl-<redacted>",
    "object":"chat.completion",
    "created":1756330925,
    "model":"./Mistral-7B-Instruct-v0.2-GPTQ",
    "choices":[
      {
        "index":0,
        "message":{
          "role":"assistant",
          "content":" Hello, I'm here to help! Here are five words: \"Hello, welcome friend.\"",
          "refusal":null,
          "annotations":null,
          "audio":null,
          "function_call":null,
          "tool_calls":[],
          "reasoning_content":null
        },
        "logprobs":null,
        "finish_reason":"stop",
        "stop_reason":null
      }
    ],
    "service_tier":null,
    "system_fingerprint":null,
    "usage":{
      "prompt_tokens":16,
      "total_tokens":37,"
    }
  }
</pre>
The server returned a JSON object containing:
- A unique request ID (chatcmpl-...).
- Metadata such as the object type, creation timestamp, and model ID.
- The generated output from the model inside the choices array.
- Token usage information showing how many tokens were processed.

This shows that:
- The model was successfully served by vLLM
- The API correctly responded to a chat request
- The Mistral-7B model produced a valid text output


<br>

## Errors encountered
### Error #1: Wrong dtype with GPTQ

The server was started with this command:
<pre>
  python3 -m vllm.entrypoints.openai.api_server
--model ./Mistral-7B-Instruct-v0.2-GPTQ \
--dtype auto \
--quantization gptq \
--port 8000
</pre>
This resulted in the following error: 
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
- “Header too large” usually happens when the model weights didn’t download correctly — often because of Git LFS (Large File Storage... Explained below).

The issue can be resolved by removing Mistral-7B and reinstalling it. The -rf flags tell rm to delete the directory and all of its contents recursively, and to do so forcefully without prompts. This ensures that no leftover or potentially corrupted files remain.
<pre>
  rm -rf Mistral-7B-Instruct-v0.2-GPTQ
  git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
</pre>

Before moving on to the next step and entering the Mistral-7B file directory, Ensure that Git LFS is installed in the virtual environment:
<pre>
  sudo apt install git-lfs
  git lfs install
</pre>

After Git LFS is installed, move into the Mistral-7B file directory and run "Git LFS pull"
<pre>
  cd Mistral-7B-Instruct-v0.2-GPTQ
  git LFS pull
</pre>
- Hugging Face and GitHub use Git LFS for big files (like multi-GB model weights). If you clone a repo with big files without Git LFS, you only get pointer stubs (tiny text files, not the full model).
- When vLLM tries to load those stubs as real model weights → safetensors chokes → error
- Doing this process, Git will actually fetch the multi-GB .safetensors files instead of stubs and you should be able to proceed without error. 
