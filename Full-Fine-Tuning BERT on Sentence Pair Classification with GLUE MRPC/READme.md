# 🧠 FULL BERT Fine-Tuning on GLUE MRPC Dataset - Full Walkthrough

This project demonstrates end-to-end fine-tuning of a BERT model on the GLUE MRPC (Microsoft Research Paraphrase Corpus) dataset using the Hugging Face Transformers and Datasets libraries. It also covers manual and Accelerate-powered training loops.

---

## 📂 Dataset and Tokenization

### 🔹 Dataset Used: `glue`, subset: `mrpc`

The MRPC dataset contains pairs of sentences and a label indicating whether they are paraphrases.

```python
raw_datasets = load_dataset('glue', "mrpc")
```

### 🔹 Tokenizer Initialization

We use the pre-trained BERT tokenizer from the Hugging Face model hub.

```python
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

### 🔹 Tokenization Function

Tokenizes sentence pairs with truncation.

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example['sentence2'], truncation=True)
```

### 🔹 Batched Tokenization & Dynamic Padding

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

---

## 🧹 Data Preprocessing

```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
```

This is necessary to prepare the dataset for PyTorch Dataloaders.

---

## 📦 Dataloaders

```python
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator)
```

---

## 🧠 Model Setup

### 🔹 Load Pre-trained BERT

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

### 🔹 Optimizer and Scheduler

```python
optimizer = AdamW(model.parameters(), lr=5e-5)

lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_epochs * len(train_dataloader)
)
```

---

## 🔁 Manual Training Loop

```python
model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

This allows custom control over how training proceeds.

---

## 📊 Evaluation

### 🔹 Using `evaluate` Library

```python
metric = evaluate.load("glue", "mrpc")

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = torch.argmax(outputs.logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

---

## 🚀 Accelerate-Enhanced Training

### 🔹 Initialize Accelerator

```python
accelerator = Accelerator()
```

### 🔹 Prepare all components

```python
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)
```

### 🔹 Training Loop

```python
model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

`Accelerator` automatically handles device placement, gradient accumulation, mixed-precision, and distributed training.

---

## 💡 Concepts Covered (In-Depth)

### ✅ `AutoTokenizer` and `AutoModel`

* Abstracts away model/tokenizer type.
* Easy to swap out model architectures.

### ✅ Data Collation

* `DataCollatorWithPadding` enables dynamic padding during batching.

### ✅ TrainingArguments vs Manual Loop

* Manual loop gives control.
* Trainer API automates logging, checkpointing, etc.

### ✅ Learning Rate Scheduler

* `linear` scheduler used here reduces learning rate linearly from the initial value to 0.

### ✅ Evaluation using `evaluate`

* GLUE metrics are task-specific and standardized.

### ✅ Accelerate

* Simplifies device management and optimization.

---

## 🧪 Output Shapes & Inspection

```python
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)  # torch.Size([8, 2])
```

* Output logits are for 2-class classification.

---

## 📈 Performance Monitoring

Use `tqdm` progress bar for visual feedback during training.

```python
progress_bar = tqdm(range(num_training_steps))
```

---

