# 🧠 Fine-Tuning BERT on Sentence Pair Classification with GLUE MRPC - Interview Prep

---

## 📌 Basic & Intermediate-Level Questions

### ❓ What is the use of `DataCollatorWithPadding`?

**Answer:** It dynamically pads each batch to the length of the longest sequence in that batch, minimizing computation waste and avoiding padding all sequences to a static max length.

---

### ❓ Explain the difference between static and dynamic padding.

**Answer:**

* Static padding uses a fixed maximum length for all sequences, leading to unnecessary padding.
* Dynamic padding only pads to the longest example in a batch, improving efficiency.

---

### ❓ What does `np.argmax(logits, axis=-1)` do?

**Answer:** It selects the index of the highest value in the logits array, converting raw model output into class predictions.

---

### ❓ Why use `AutoTokenizer` over `BertTokenizer`?

**Answer:** `AutoTokenizer` allows you to use any model architecture by just changing the checkpoint string, improving modularity and reusability of code.

---

### ❓ What’s the `processing_class=tokenizer` in the `Trainer`?

**Answer:** The `processing_class` argument allows the Trainer to automatically handle input data processing, including tokenization and feature extraction..

---

### ❓ How does Hugging Face `Trainer` simplify training?

**Answer:** It abstracts away training loops, evaluation, logging, checkpointing, mixed-precision training, and distributed training, making development much easier.

---

### ❓ How would you adapt this for multi-class classification?

**Answer:**

* Set `num_labels=n` (where n is number of classes).
* Make sure your metric function and label mapping reflect multi-class logic.

---

## 🔍 Advanced-Level Questions

### ❓ What is the difference between `Trainer`, `Accelerate`, and `PyTorch Lightning`?

**Answer:**

* `Trainer`: Easy-to-use API for NLP fine-tuning (best for quick experiments).
* `Accelerate`: Lower-level API for device placement and distributed training.
* `PyTorch Lightning`: General-purpose training abstraction for reproducible research.

---

### ❓ Why use `batched=True` in `.map()`?

**Answer:** It allows you to process a batch of examples at once, significantly improving tokenization performance.

---

### ❓ How is the \[CLS] token used in BERT classification?

**Answer:** BERT’s final hidden state corresponding to the `[CLS]` token is used as a sentence-level embedding and fed into a linear classifier for prediction.

---

### ❓ What happens if you forget `truncation=True`?

**Answer:** If any sentence pair exceeds the model's maximum token length (usually 512), you’ll get an error during inference or training.

---

### ❓ How does `evaluate` differ from `sklearn.metrics`?

**Answer:** `evaluate` offers task-specific metrics aligned with datasets like GLUE, and integrates well with Hugging Face's ecosystem. `sklearn.metrics` provides more general-purpose ML metrics.

---

### ❓ Why is `num_labels=2` important?

**Answer:** This tells the model to create a classification head with 2 output units for binary classification. If omitted or set incorrectly, you'll get mismatched shape errors.

---

### ❓ What happens internally when calling `trainer.train()`?

**Answer:** It loads the model and data to GPU, iterates through the dataset using a `DataLoader`, computes loss, backpropagates, updates weights, logs metrics, and saves checkpoints.

---

### ❓ How can you prevent overfitting?

**Answer:**

* Early stopping
* Dropout layers
* Weight decay
* Use proper learning rate (2e-5 to 5e-5 for BERT)
* Fewer training epochs if dataset is small

---

### ❓ Can you change the tokenizer and model independently?

**Answer:** No. Tokenizer and model should be from the same family. Mismatch leads to vocabulary mismatches and poor performance or runtime errors.

---

### ❓ How do you visualize training progress?

**Answer:**
Set `logging_dir="./logs"` and run:

```bash
tensorboard --logdir=./logs
```

This visualizes metrics like loss, accuracy, and learning rate over time.

---

## 📌 Bonus: Code Debugging Questions

### ❓ What’s a common mistake when using the `Trainer` API with tokenizers?

**Answer:** Using `processing_class` instead of `tokenizer` or forgetting to pass `compute_metrics` if custom metrics are required.

---

### ❓ If evaluation metric shows `0.0` accuracy, what would you check?

**Answer:**

* Whether logits were converted to predictions using `argmax`
* Whether label ids are correctly passed
* If dataset was shuffled or corrupted
* Check `num_labels` and final model layer shape

---
