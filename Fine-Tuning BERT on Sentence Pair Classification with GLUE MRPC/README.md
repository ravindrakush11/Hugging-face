# 🧠 Fine-Tuning BERT on Sentence Pair Classification with GLUE MRPC

This project demonstrates **sentence pair classification** on the [GLUE MRPC dataset](https://huggingface.co/datasets/glue/viewer/mrpc) using **BERT (bert-base-uncased)** and **Hugging Face Transformers**. The goal is to predict whether a pair of sentences are semantically equivalent.

---

## 📌 Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Setup Instructions](#setup-instructions)
* [Code Walkthrough](#code-walkthrough)
* [Training & Evaluation](#training--evaluation)
* [GPU Availability](#gpu-availability)
* [Compute Metrics Function](#compute-metrics-function)
* [Output Example](#output-example)
* [In-Depth Breakdown](#in-depth-breakdown-of-hard-parts)
* [Interview Questions](#interview-questions)
* [Dependencies](#dependencies)
* [References](#references)

---

## 🔍 Project Overview

This code fine-tunes the BERT model on the **Microsoft Research Paraphrase Corpus (MRPC)** from the **GLUE benchmark** using Hugging Face's `Trainer` API. It includes:

* Data loading and tokenization
* Dynamic padding and batching
* Model initialization with classification head
* Training, evaluation, and metric computation

---

## 📚 Dataset

* **Name:** GLUE MRPC
* **Task:** Sentence Pair Classification (Are two sentences paraphrases?)
* **Labels:** Binary (1 = paraphrase, 0 = not paraphrase)

---

## 🧠 Model Architecture

* **Base Model:** `bert-base-uncased`
* **Head:** Linear layer for 2-class classification
* **Loss Function:** CrossEntropyLoss
* **Training Framework:** Hugging Face `Trainer`

---

## ⚙️ Setup Instructions

1. **Install dependencies**

   ```bash
   pip install transformers datasets evaluate torch
   ```

2. **Run the training script**

   ```bash
   python app.py
   ```

---

## 🧾 Code Walkthrough

### ✅ Dataset Loading and Tokenization

```python
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

### ✅ Data Collation for Dynamic Padding

```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### ✅ Model and Trainer Setup

```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,  # Fixed typo here
)
```

### ✅ Train and Evaluate

```python
trainer.train()

predictions = trainer.predict(tokenized_datasets['validation'])
preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

---

## ⚡ GPU Availability

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## 📊 Compute Metrics Function (Best Practice)

```python
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
```

---

## ✅ Output Example

You can expect an output like:

```
(172, 2) (172,)
{'accuracy': 0.847, 'f1': 0.899}
```

---

## 🔬 In-Depth Breakdown

### 1. `dataset.map(..., batched=True)`

* Enables batch-wise transformation (tokenization), improving speed.
* Often misunderstood with Python's map function — this is from Hugging Face Datasets.

---

### 2. `DataCollatorWithPadding`

* Automatically pads examples dynamically in each batch.
* Avoids the inefficiency of static padding to max sequence length (e.g., 512).

---

### 3. `AutoModelForSequenceClassification.from_pretrained(...)`

* Loads BERT with a classification head.
* Internally: `Linear(hidden_size, num_labels)` applied to the `[CLS]` token.

---

### 4. `compute_metrics(...)`

* Converts logits to class predictions with `argmax`.
* Uses GLUE’s standard metric evaluator.
* Cleaner and modular for multiple evaluations during training.

---

## 🧠 Interview Questions

### ❓ What is the use of `DataCollatorWithPadding`?

➡ Dynamically pads each batch to the length of the longest sequence in the batch, reducing wasted computation and memory.

---

### ❓ Explain the difference between static and dynamic padding.

➡ Static pads all sequences to a fixed length (inefficient). Dynamic pads only to the longest sequence in each batch (used here).

---

### ❓ What does `np.argmax(logits, axis=-1)` do?

➡ Converts model logits into discrete class predictions (index of max value across the class dimension).

---

### ❓ Why use `AutoTokenizer` over `BertTokenizer`?

➡ `AutoTokenizer` is checkpoint-agnostic and works with any transformer model type, improving code reusability.

---

### ❓ What’s wrong with `processing_class=tokenizer` in the `Trainer`?

➡ `processing_class` is not a valid argument. It should be `tokenizer=tokenizer`.

---

### ❓ How does Hugging Face `Trainer` simplify training?

➡ Handles training loop, evaluation, logging, saving checkpoints, gradient accumulation, etc., abstracting boilerplate code.

---

### ❓ How would you adapt this for multi-class classification?

➡ Set `num_labels=n` in `AutoModelForSequenceClassification` and ensure metric function handles multi-class metrics like accuracy or precision/recall.

---

## 📦 Dependencies

```bash
pip install transformers datasets evaluate torch numpy
```

---

## 📚 References

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [GLUE Benchmark](https://gluebenchmark.com/)
* [MRPC Dataset](https://huggingface.co/datasets/glue/viewer/mrpc)
* [Hugging Face Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
