# 🚀 **vLLM API Server**  
**(OpenAI-Compatible)**

Welcome to the **vLLM API Server**, a **fast** and **memory-efficient** solution for running OpenAI-compatible models using [vLLM](https://github.com/vllm-project/vllm). With support for popular models like `facebook/opt-1.3b` and `mistralai/Mistral-7B-Instruct-v0.1`, this server can be easily deployed on **WSL 2** with **GPU support**. Experience the future of AI model serving with seamless performance!

---

## ✅ **System Requirements**

Before running the vLLM server, make sure you have the following:

- **🖥️ Windows 11** (for a smooth experience)
- **🐧 WSL 2 with Ubuntu** (preferably 22.04 or 24.04)
- **💻 NVIDIA GPU** with **WSL2-compatible CUDA drivers**
- **🐍 Python 3.10+** (or later versions)
- **🌐 Stable Internet Connection** (for downloading models)

---

## ⚙️ **Installation Guide (via WSL 2)**

Here’s a step-by-step guide to setting up the **vLLM API Server** on **WSL 2**:

### 1. **Launch WSL (Ubuntu)**  
First, open the **WSL terminal** to start the setup process.

### 2. **Update Your System & Install Dependencies**  
Run the following commands to update your system and install the necessary packages:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y
```


### 3. **Create and activate a Python virtual environment**

This step is optional but recommended for managing dependencies.

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
```

### 4. **Install vLLM**

vLLM contains pre-compiled C++ and CUDA (12.6) binaries, so you don't need to install PyTorch beforehand to avoid dependency conflicts.

```bash
pip install vllm
```

---

## 🚀 **Run the vLLM API Server**

To start the API server, use the following command:

```bash
python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-1.3b
```

Once the server starts, it will be available at:

* **API URL:** `http://localhost:8000/v1/completions`
* **Model location:** `~/.cache/huggingface`

---

## 📡 **Test the API with Postman**

### Request Configuration:

* **Method:** `POST`
* **URL:** `http://localhost:8000/v1/completions`
* **Headers:**

  * `Content-Type: application/json`

### Request Body (raw, JSON):

```json
{
  "model": "facebook/opt-1.3b",
  "prompt": "Hello, world. What is vLLM?",
  "max_tokens": 50
}
```

Click **Send**, and you should receive a response similar to:

```json
{
    "id": "cmpl-084e842947374e8582b6d760df5a95d8",
    "object": "text_completion",
    "created": 1746636833,
    "model": "facebook/opt-1.3b",
    "choices": [
        {
            "index": 0,
            "text": " The latter part of this post is about the production “elegance” passed down through this blog that I learned about this summer and from which no additional explanation here is necessary. vLLM traits were originally selected in the spring, they",
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null,
            "prompt_logprobs": null
        }
    ],
    "usage": {
        "prompt_tokens": 11,
        "total_tokens": 61,
        "completion_tokens": 50,
        "prompt_tokens_details": null
    }
}
```

---

## 🧹 **Cleanup and Uninstallation**

To remove **vLLM** and its dependencies, follow these steps:

### 1. **Uninstall the packages:**

```bash
pip uninstall vllm torch torchvision torchaudio
```

### 2. **Optional: Remove downloaded models**

```bash
rm -rf ~/.cache/huggingface
```

### 3. **Optional: Delete the virtual environment**

```bash
deactivate
rm -rf vllm-env
```

---

## 📚 **Useful Resources**

* **vLLM GitHub Repository:**
  [vLLM GitHub](https://github.com/vllm-project/vllm)

* **CUDA WSL Guide for GPU:**
  [NVIDIA CUDA WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)

---

**Enjoy your fast and efficient AI model deployment! 🚀**
