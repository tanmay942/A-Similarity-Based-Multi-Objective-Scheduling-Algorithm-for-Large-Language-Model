# A-Similarity-Based-Multi-Objective-Scheduling-Algorithm-for-Large-Language-Model

## Overview

SMS-LLM is a similarity-based adaptive scheduling framework for selecting the most suitable Large Language Model (LLM) for a given prompt.

The system dynamically selects the best LLM based on:

* Latency
* Response Quality
* Token Consumption

Instead of using a single static model, SMS-LLM uses semantic similarity between prompts and historical LLM performance data to optimize model selection.

---

## Features

* Adaptive Multi-LLM Scheduling
* Semantic Similarity-Based Prompt Matching
* Embedding-Based Query Analysis
* Latency-Oriented Scheduling (LS-LLM)
* Quality-Oriented Scheduling (QS-LLM)
* Multi-Objective Optimization
* LLM-as-a-Judge Evaluation
* Token Usage Optimization

---

## Models Used

* LLaMA 3
* Mistral
* Qwen2.5

---

## Architecture

### Offline Phase

* Generate prompt embeddings
* Execute prompts on multiple LLMs
* Store latency, quality score, and token count
* Build knowledge base

### Online Phase

* Compute query embedding
* Retrieve similar prompts
* Estimate LLM performance
* Select optimal LLM dynamically

---

## Tech Stack

* Python
* Ollama
* Sentence Embeddings
* LLM-as-a-Judge
* NumPy
* Pandas

---

## Performance Highlights

* Up to 29.1% latency reduction
* Up to 5.76% quality improvement
* Up to 25.3% token reduction

Compared against:

* Static single-model deployment
* Random scheduling baseline

---

## Project Structure

```bash
SMS-LLM/
│
├── datasets/
├── embeddings/
├── scheduler/
├── evaluation/
├── results/
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone <repo-link>

cd SMS-LLM

pip install -r requirements.txt

python main.py
```

---

## Research Paper

This project was developed as a B.Tech Final Year Project at:

National Institute of Technology (NIT) Warangal

Under the guidance of:
Prof. Sanjaya Kumar Panda

---

## Authors

* Tanmay Soni
* Vikram

---

## Future Work

* Dynamic weight tuning
* Cost-aware scheduling
* Real-time adaptive learning
* Distributed LLM scheduling
