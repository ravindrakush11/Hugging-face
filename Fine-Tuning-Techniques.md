# 🌟 LoRA (Low-Rank Adaptation)

Fine-tuning large language models is a **resource-intensive process**.  
**LoRA** is a technique that allows us to fine-tune large models with **a small number of parameters**.

It works by **adding and optimizing smaller matrices** to the attention weights, typically **reducing trainable parameters by about 90%**.

---

## 🔍 Understanding LoRA

**LoRA (Low-Rank Adaptation)** is a **parameter-efficient fine-tuning** technique that:
- **Freezes the pre-trained model weights**
- **Injects trainable rank decomposition matrices** into the model’s layers

Instead of training all parameters, LoRA:
- **Decomposes weight updates into smaller matrices** via low-rank decomposition
- **Reduces trainable parameters significantly** while maintaining performance

> 💡 *Example:* On **GPT-3 175B**, LoRA:
> - Reduced trainable parameters by **10,000x**
> - Cut GPU memory requirements by **3x** compared to full fine-tuning

During **inference**, adapter weights:
- Can be **merged with the base model**
- **Do not add latency overhead**

**LoRA is ideal** for adapting large models to specific tasks/domains with **lower compute/memory cost**.

---

## ✅ Key Advantages of LoRA

### 🧠 Memory Efficiency
- Only **adapter parameters** are stored in GPU memory
- **Base model weights** are frozen (can use lower precision)
- Enables fine-tuning **large models on consumer GPUs**

### ⚙️ Training Features
- **Native PEFT/LoRA integration** with minimal setup
- Support for **QLoRA (Quantized LoRA)** for better memory efficiency

### 🗂️ Adapter Management
- **Adapter weight saving** during checkpoints
- Features to **merge adapters back** into the base model

---

## 📦 Loading LoRA Adapters with PEFT

**PEFT** is a library that offers a **unified interface** to load and manage **PEFT methods** including LoRA.

It allows:
- **Easy switching** between PEFT methods
- **Simple experimentation** with fine-tuning strategies

> We’ll use the `LoRAConfig` class from PEFT in our example.

### 🛠️ Setup Steps
1. **Define** the LoRA configuration (rank, alpha, dropout)
2. **Create** the `SFTTrainer` with PEFT config
3. **Train and save** the adapter weights

---

## ⚙️ LoRA Configuration Parameters

| **Parameter**       | **Description** |
|---------------------|-----------------|
| `r` (rank)          | Dimension of low-rank matrices (typically **4–32**). Lower = more compression. |
| `lora_alpha`        | **Scaling factor** for LoRA layers (usually **2x rank**). Higher = stronger adaptation. |
| `lora_dropout`      | **Dropout probability** for LoRA layers (**0.05–0.1**). Helps avoid overfitting. |
| `bias`              | Controls training of bias terms. Options: `"none"`, `"all"`, or `"lora_only"`. Most common: `"none"`. |
| `target_modules`    | Specifies which modules to apply LoRA to (e.g., `"all-linear"`, `"q_proj,v_proj"`). More modules = more adaptability. |

---
