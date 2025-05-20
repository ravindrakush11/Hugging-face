## 🤖 Hugging Face NLP Model Classes — Interview Q&A Guide

### 1. `AutoModel`

**Q: What is `AutoModel` used for?**  
A: It loads the base transformer model without any task-specific head. It is typically used for feature extraction or embedding.

**Q: When would you choose `AutoModel` over task-specific classes?**  
A: When you need hidden states or embeddings (e.g., for clustering, semantic similarity) rather than task-specific outputs.

---

### 2. `AutoModelForCausalLM`

**Q: What is the primary use case for `AutoModelForCausalLM`?**  
A: Text generation via causal language modeling (e.g., GPT, LLaMA).

**Q: Can you use `.generate()` with this model?**  
A: Yes. It supports greedy, beam, and sampling-based generation.

---

### 3. `AutoModelForSeq2SeqLM`

**Q: When do you use `AutoModelForSeq2SeqLM`?**  
A: For sequence-to-sequence tasks like translation, summarization, and dialogue generation.

**Q: What models are commonly loaded with this?**  
A: T5, BART, mBART, MarianMT, etc.

---

### 4. `AutoModelForMaskedLM`

**Q: What is `AutoModelForMaskedLM` used for?**  
A: Masked Language Modeling tasks (e.g., pretraining or fine-tuning models like BERT).

**Q: How does it differ from `AutoModelForCausalLM`?**  
A: `AutoModelForMaskedLM` predicts missing tokens using context on both sides, whereas `AutoModelForCausalLM` predicts the next token using left context only.

---

### 5. `AutoModelForSequenceClassification`

**Q: What's the purpose of this class?**  
A: Adds a linear classification head for text classification tasks (e.g., sentiment analysis, spam detection).

**Q: How to fine-tune it for binary classification?**  
A: Set `num_labels=2` when loading the model or in the config.

---

### 6. `AutoModelForTokenClassification`

**Q: What tasks use this model class?**  
A: Named Entity Recognition (NER), Part-of-Speech tagging, and other token-level classification.

**Q: What's the output structure?**  
A: A classification score for each token.

---

### 7. `AutoModelForQuestionAnswering`

**Q: What's the use case?**  
A: Extractive question answering, where the model predicts the start and end positions of the answer span in a context.

**Q: Example model?**  
A: BERT fine-tuned on SQuAD.

---

### 8. `AutoModelForMultipleChoice`

**Q: Where is this model used?**  
A: In multiple-choice tasks like SWAG or CommonsenseQA.

**Q: What is the input format?**  
A: Each choice is treated as a separate input, and the model outputs scores for each.

---

### 9. `AutoModelForNextSentencePrediction`

**Q: What is its main purpose?**  
A: To determine if a given sentence B logically follows sentence A — part of BERT’s original pretraining objective.

**Q: Is it widely used now?**  
A: No, it’s largely considered obsolete in favor of next-token prediction and contrastive methods.

---

### 10. `AutoModelForPreTraining`

**Q: When should you use this class?**  
A: When doing pretraining tasks combining MLM, NSP, or other custom objectives.

**Q: Example scenario?**  
A: Training BERT from scratch on a new domain.
