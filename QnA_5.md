## 🔧 **LoRA / PEFT (Parameter-Efficient Fine-Tuning)**

---

### ✅ What is PEFT?

**Parameter-Efficient Fine-Tuning (PEFT)** refers to a set of techniques that **fine-tune only a small part** of a large model instead of the entire parameter set. It’s ideal for:

* Large models (BLOOM, LLaMA, Falcon, etc.)
* Resource-constrained environments (low GPU RAM)
* Domain-specific or task-specific adaptation

---

### ✅ What is LoRA?

**LoRA (Low-Rank Adaptation)** is the most popular PEFT method. Instead of fine-tuning the full weight matrices (millions/billions of parameters), LoRA **adds small trainable matrices (rank `r`)** to selected layers and **freezes the base model**.

It introduces **two low-rank matrices (A and B)** such that:

$$
W' = W + \Delta W \quad \text{where} \quad \Delta W = A \cdot B
$$

---

### ✅ Benefits of LoRA

| Feature                   | Benefit                                                     |
| ------------------------- | ----------------------------------------------------------- |
| 🧠 Fewer Trainable Params | Only a few million parameters                               |
| 🚀 Faster Training        | Smaller gradient updates                                    |
| 🧊 Lower Memory           | Doesn't update the entire model                             |
| 🔄 Reusable               | Can switch tasks quickly by loading different LoRA adapters |
| 🌐 Community-ready        | Integrated with Hugging Face `peft` library                 |


---

### 🔧 Key Hyperparameters in LoRA

| Hyperparameter   | Description                                               | Example                |
| ---------------- | --------------------------------------------------------- | ---------------------- |
| `r`              | Rank of the low-rank update matrices A and B              | `8`, `16`              |
| `lora_alpha`     | Scaling factor for the low-rank matrices                  | `16`, `32`             |
| `lora_dropout`   | Dropout probability for LoRA layers                       | `0.05`, `0.1`          |
| `bias`           | Whether to train bias params (`none`, `all`, `lora_only`) | `"none"`               |
| `target_modules` | Model modules to apply LoRA (e.g., attention projections) | `["q_proj", "v_proj"]` |
| `task_type`      | Defines type of task (`CAUSAL_LM`, `SEQ_CLS`, etc.)       | `TaskType.CAUSAL_LM`   |

---

### 📦 Hugging Face `peft` Library Tasks

| Task Type            | Use Case                           |
| -------------------- | ---------------------------------- |
| `CAUSAL_LM`          | Text generation (e.g., GPT, LLaMA) |
| `SEQ_CLS`            | Sequence classification            |
| `TOKEN_CLS`          | NER or token tagging               |
| `SEQ_2_SEQ_LM`       | Summarization, translation         |
| `QUESTION_ANS`       | Extractive QA                      |
| `FEATURE_EXTRACTION` | Embedding extraction               |

---

### 🧠 Under the Hood

LoRA modifies **only certain weights** like `query`, `key`, `value` in attention layers. Instead of updating the full 7B+ parameters, LoRA might only update \~5M parameters (a 1000x reduction in training size).

---

### 📚 Example Use Cases

* Fine-tune **LLaMA2** on **medical Q\&A** with LoRA and just 1 GPU.
* Use LoRA for **multi-lingual adaptation** of a GPT model.
* Plug in LoRA adapters into an **8-bit quantized model** (via `bitsandbytes`).

---

### 📎 Tools that Support LoRA

| Tool              | Role                               |
| ----------------- | ---------------------------------- |
| 🤗 `peft`         | Core PEFT library                  |
| 🤗 `transformers` | Model loading and integration      |
| `bitsandbytes`    | 8-bit / 4-bit quantization         |
| `accelerate`      | Multi-GPU and optimization support |
| `wandb`           | Logging and visualization          |

---

### 📌 Best Practices

* Use `load_in_8bit=True` to conserve GPU RAM.
* Combine LoRA with **gradient checkpointing** for large models.
* Tune `r` and `lora_alpha` for balance between speed and accuracy.

---

---


---

Here’s a **concise yet detailed breakdown** of the top **parameter-efficient fine-tuning (PEFT)** methods used in Large Language Models (LLMs), including:

> **PEFT · LoRA · QLoRA · Adapters · Prompt Tuning**

These are **complementary**, not mutually exclusive—often used based on compute budget, task size, or deployment needs.

---

## 🧠 1. **PEFT (Parameter-Efficient Fine-Tuning)** – *The umbrella concept*

### 🔹 Definition:

A collection of techniques that **fine-tune only a small subset of model parameters** (or add a few trainable ones) instead of updating the full model.

### 🔹 Why PEFT?

* Reduce compute and memory requirements
* Faster training on consumer-grade hardware (1 GPU or even CPU)
* Easier to store/share adapters (\~MBs vs GBs)

### 🔹 Types:

* LoRA
* QLoRA
* Adapters
* Prefix/Prompt tuning
* BitFit, IA3, etc.

---

## 🔧 2. **LoRA (Low-Rank Adaptation)**

### 🔹 Principle:

Instead of updating large matrices, **add two small trainable matrices (A & B)** of rank `r` to certain layers:

$$
\Delta W = A \cdot B
$$

### 🔹 Characteristics:

| Feature               | Value                         |
| --------------------- | ----------------------------- |
| Trainable Params      | \~0.1–2%                      |
| Memory Efficient      | ✅                             |
| Use with 8-bit models | ✅                             |
| Popular With          | Causal LM, Seq2Seq, NER, etc. |
| HF Tool               | `peft.LoraConfig`             |

### 🔹 Hyperparameters:

* `r`: Rank of LoRA (e.g. `8`, `16`)
* `lora_alpha`: Scaling factor
* `lora_dropout`: Dropout on LoRA layers
* `target_modules`: Layers to apply (e.g. `q_proj`, `v_proj`)

---

## ⚡ 3. **QLoRA (Quantized LoRA)**

### 🔹 Principle:

Combines **4-bit quantized models** with LoRA adapters.

$$
\text{Quantize} \rightarrow \text{LoRA fine-tune only adapters}
$$

### 🔹 Benefits:

| Feature     | Value                                  |
| ----------- | -------------------------------------- |
| Model Size  | 4x smaller (e.g., 65B → 16GB)          |
| RAM Usage   | Fits on a single 24GB GPU              |
| Performance | Comparable to full finetuning          |
| Tools       | `bitsandbytes`, `peft`, `transformers` |

### 🔹 Typical Stack:

* Load model with `load_in_4bit=True` (via `bitsandbytes`)
* Use `bnb_4bit_compute_dtype='bfloat16'`
* Add `LoRA` via `peft`

---

## ⚙️ 4. **Adapters**

### 🔹 Principle:

Inject **small bottleneck layers (adapter blocks)** inside the transformer layers, and fine-tune **only the adapters**.

### 🔹 Anatomy of Adapter:

```
LayerNorm → DownProj → Nonlinearity → UpProj → Residual Add
```

### 🔹 Characteristics:

| Feature                  | Value                                         |
| ------------------------ | --------------------------------------------- |
| Trainable Params         | \~1–2%                                        |
| Modular Plug & Play      | ✅                                             |
| Can be used sequentially | ✅                                             |
| Tasks                    | Classification, QA, NER, generation           |
| HF Tool                  | `adapter-transformers` (separate from `peft`) |

---

## ✏️ 5. **Prompt Tuning / Prefix Tuning**

### 🔹 Prompt Tuning:

* Learnable **token embeddings** prepended to the input
* Rest of the model remains **frozen**
* Smallest number of trainable params

### 🔹 Prefix Tuning:

* Learns **prefix vectors per attention layer**
* Injects soft prompts at each layer, not just input

| Method        | Params     | Trainable             | Quality |
| ------------- | ---------- | --------------------- | ------- |
| Prompt Tuning | \~few KB   | Embeddings only       | Medium  |
| Prefix Tuning | \~0.1–0.5% | Attention keys/values | High    |
| LoRA          | \~0.5–2%   | Q/V layers or FFN     | Higher  |

### 🔹 Ideal For:

* Training with **very small data**
* Few-shot or zero-shot prompt augmentation
* When **storage size matters**

---

## 📊 Summary Table

| Method            | % Params Tuned | Uses Base Model | Needs Quantization | Ideal For                             |
| ----------------- | -------------- | --------------- | ------------------ | ------------------------------------- |
| **LoRA**          | 0.5% – 2%      | ✅               | Optional           | Most common PEFT                      |
| **QLoRA**         | 0.5% – 2%      | 4-bit model     | ✅                  | Very large models (e.g., 33B, 65B)    |
| **Adapters**      | \~2%           | ✅               | ❌ (or optional)    | Modular tasks, multi-domain           |
| **Prompt Tuning** | <0.1%          | ✅               | ❌                  | Extremely low-resource fine-tuning    |
| **Prefix Tuning** | 0.1–0.5%       | ✅               | ❌                  | Better performance than prompt tuning |

---

## 🧰 Tools Used Across Methods

| Tool                   | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| 🤗 `peft`              | Official Hugging Face library for LoRA, Prefix, Prompt |
| `transformers`         | Base model loading and Trainer                         |
| `bitsandbytes`         | Quantization (for QLoRA)                               |
| `adapter-transformers` | Separate lib for Adapter-based fine-tuning             |
| `accelerate`           | Easy multi-GPU & mixed precision setup                 |

---

