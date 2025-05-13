# 🧠 BERT Fine-Tuning on MRPC - Interview Q\&A (Basic to Advanced)

This document contains interview-style questions and detailed answers based on the end-to-end BERT fine-tuning code using the GLUE MRPC dataset.

---

## 🔰 Basic Level

### 1. **What is the MRPC dataset used for?**

**Answer:** MRPC (Microsoft Research Paraphrase Corpus) is a binary classification dataset that contains pairs of sentences. The task is to predict whether the two sentences are paraphrases (i.e., semantically equivalent).

### 2. **Which tokenizer is used in this project?**

**Answer:** The `AutoTokenizer` class from Hugging Face is used with the model checkpoint `bert-base-uncased`.

### 3. **What is the role of `tokenize_function`?**

**Answer:** It takes a dictionary with `sentence1` and `sentence2` and applies the tokenizer with truncation to fit within the model’s input length.

### 4. **Why do we use `batched=True` in `.map()`?**

**Answer:** It allows the tokenizer to process multiple examples at once, which is more efficient than processing one sample at a time.

### 5. **What does `DataCollatorWithPadding` do?**

**Answer:** It pads each batch dynamically during training so that all sequences have the same length in a batch, improving memory efficiency.

---

## ⚙️ Intermediate Level

### 6. **Why do we remove the columns `sentence1`, `sentence2`, and `idx`?**

**Answer:** These columns are not required by the model. Removing them saves memory and ensures the DataLoader returns only tensors needed by the model.

### 7. **Why do we rename the `label` column to `labels`?**

**Answer:** The model expects a key named `labels` in the input dictionary for loss computation. This aligns the dataset with the model’s input format.

### 8. **What is `AutoModelForSequenceClassification` used for?**

**Answer:** It loads a BERT model with a classification head appropriate for sequence pair classification tasks (e.g., binary classification in MRPC).

### 9. **What is the shape of model outputs?**

**Answer:** `logits` have shape `[batch_size, num_labels]`, where `num_labels=2` for MRPC. `loss` is a scalar tensor.

### 10. **Why do we use `AdamW` as optimizer?**

**Answer:** `AdamW` (Adam with weight decay) helps prevent overfitting and is commonly used for fine-tuning Transformers.

### 11. **What does `tokenized_datasets.set_format("torch")` do?**

**Answer:** It converts the dataset columns into PyTorch tensors, which are required for training with PyTorch’s `DataLoader` and models.

---

## 🚀 Advanced Level

### 12. **Explain the linear learning rate scheduler.**

**Answer:** `get_scheduler("linear")` linearly decreases the learning rate from an initial value to zero over the number of training steps. This helps stabilize training.

### 13. **Why use `accelerator.prepare()`?**

**Answer:** It handles device placement (CPU/GPU), mixed precision training, and prepares models and data loaders for distributed or efficient training.

### 14. **How is gradient accumulation handled with Accelerate?**

**Answer:** `accelerator.backward(loss)` replaces `loss.backward()` and manages gradient accumulation internally, if configured.

### 15. **What is the purpose of `metric.add_batch` and `metric.compute()`?**

**Answer:** These functions compute evaluation metrics (accuracy, F1, etc.) by collecting predictions and ground-truth labels over the entire evaluation dataset.

### 16. **What are potential issues if you don’t set `.set_format("torch")`?**

**Answer:** Data returned from the dataset won’t be tensors, causing runtime errors when using `DataLoader` or passing batches to the model.

### 17. **What are the differences between manual training loop and using `Trainer`?**

**Answer:** The `Trainer` API abstracts away the boilerplate code and is easier for rapid experimentation. Manual loops offer greater customization for research-level modifications and debugging.

### 18. **What happens if `optimizer.zero_grad()` is skipped?**

**Answer:** Gradients accumulate across steps, leading to incorrect weight updates and unstable training.

### 19. **Why might evaluation accuracy be low even if training loss decreases?**

**Answer:**

* Overfitting due to small dataset size.
* Inappropriate learning rate.
* Model not being evaluated in `eval()` mode.

### 20. **How does Accelerate improve training compared to vanilla PyTorch?**

**Answer:** Accelerate abstracts device management, supports mixed precision, makes multi-GPU and TPU training easier, and reduces boilerplate code for distributed settings.

### 21. **Why is `with torch.no_grad()` used during evaluation?**

**Answer:** It disables gradient computation to reduce memory usage and speed up inference since gradients are not needed for evaluation.

### 22. **How is batching handled efficiently in this code?**

**Answer:** Using `DataCollatorWithPadding` ensures batches are padded dynamically only to the length of the longest sample in the batch, reducing computation waste.

---

## 🧪 Debugging & Best Practices

### 23. **What should you check if training is very slow or memory errors occur?**

**Answer:**

* Check batch size and sequence length.
* Ensure dynamic padding is used.
* Use mixed precision training with Accelerate.

### 24. **How do you verify GPU usage?**

**Answer:**

```python
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### 25. **Can you use custom metrics instead of GLUE metrics?**

**Answer:** Yes, you can load or define your own metrics using `evaluate.load()` or create a custom function to compute metrics like precision, recall, or custom F1 logic.

---
