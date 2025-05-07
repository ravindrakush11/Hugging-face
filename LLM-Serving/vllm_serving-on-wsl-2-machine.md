````markdown
# 🚀 **vLLM API Server (OpenAI-Compatible)**

Welcome to the **vLLM API Server**, a fast and memory-efficient server for running OpenAI-compatible models using [vLLM](https://github.com/vllm-project/vllm). This server supports popular models like `facebook/opt-1.3b` and `mistralai/Mistral-7B-Instruct-v0.1`, and can be easily deployed on **WSL 2** with **GPU support**.

---

## ✅ **Requirements**

To run the vLLM server, ensure you have the following:

- **Windows 11**
- **WSL 2 with Ubuntu** (preferably 22.04 or 24.04)
- **NVIDIA GPU** with **WSL2-compatible CUDA drivers**
- **Python 3.10+**
- **Internet connection** (for downloading models)

---

## ⚙️ **Installation Guide (via WSL 2)**

Follow the steps below to set up the server on **WSL 2**:

### 1. **Launch WSL (Ubuntu)**

```bash
wsl
````

### 2. **Update system and install dependencies**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y
```

### 3. **Create and activate a Python virtual environment** (optional but recommended)

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
```

### 4. **Install PyTorch with CUDA support** (for GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. **Install vLLM**

```bash
pip install vllm
```

---

## 🚀 **Run the vLLM API Server**

Start the API server with the following command:

```bash
python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-1.3b
```

* The server will run on: `http://localhost:8000/v1/completions`
* The model is downloaded to: `~/.cache/huggingface`

---

## 📡 **Test the API with Postman**

Here’s how to test the running server using **Postman**:

### Request Configuration:

* **Method:** `POST`
* **URL:** `http://localhost:8000/v1/completions`
* **Headers:**

  * `Content-Type: application/json`

### Body (raw, JSON):

```json
{
  "model": "facebook/opt-1.3b",
  "prompt": "Hello, world. What is vLLM?",
  "max_tokens": 50
}
```

Click **Send** and you should receive a response like:

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

If you want to remove **vLLM** and its dependencies:

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

