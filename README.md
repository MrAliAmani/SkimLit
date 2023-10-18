# SkimLit
This project aims to extract PubMed abstracts's different parts such as methods and results.
It is based on 2 articles:
**PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts**

**Neural Networks for Joint Sentence Classification in Medical Paper Abstracts**
The baseline model is developed using a TFIDF vectorizer and multinimial naive bayse model.
The other experiments are as followed:
**Conv1D model**

**data pipeline for deep models**

**pretrained feature extraction with pretrained token embeddings using USE from tf hub.**

**Conv1D model with character embeddings and character level tokenizer**

**Combining pretrained token embeddings + character embeddings (hybrid embedding layer)**

**Transfer Learning with pretrained token embeddings + character embeddings + positional embeddings**

I have compared these different models and looked at the most wrong predictions and example predictions.**
Furtheremore, I have experimented with pretrained GloVe embeddings and TensorFlow Hub BERT PubMed expert (a language model pretrained on PubMed texts) pretrained embedding.
