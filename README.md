# 🧠 Multi-Task Customer Feedback Classifier

> A transformer-based deep learning system that classifies customer messages across 4 tasks simultaneously — sentiment, behavior score, issue category, and urgency — using a fine-tuned DistilBERT model with a shared encoder architecture and joint loss optimization.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Tasks & Labels](#tasks--labels)
- [Dataset](#dataset)
- [Model Design](#model-design)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [API Reference](#api-reference)
- [Frontends](#frontends)

---

## Overview

**Multi-Task Customer Feedback Classifier** is an end-to-end NLP pipeline that takes a raw customer message and simultaneously predicts:

- **What they feel** → Sentiment
- **How severe their tone is** → Behavior Score
- **What their message is about** → Issue Category
- **How urgently it needs attention** → Urgency Level

All four predictions come from a single forward pass through one shared DistilBERT encoder — making it efficient, consistent, and production-ready for customer support routing pipelines.

---

## Problem Statement

Customer support teams receive thousands of messages daily — complaints, billing issues, feature requests, praise, and general inquiries — all mixed together. Manually triaging these is slow and error-prone.

This system solves that by automatically:
- Detecting the emotional tone of the message
- Categorizing what the issue is about
- Flagging how urgently it needs a response
- Scoring the overall behavior quality of the interaction

All in real time, via a REST API.

---

## Architecture

```
Raw Customer Text
        │
        ▼
┌───────────────────────┐
│   Text Preprocessing   │   whitespace normalization, emoji removal
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  DistilBERT Tokenizer  │   max_length=128, padding, truncation
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  DistilBERT Encoder    │   distilbert-base-uncased
│  (Shared Backbone)     │   hidden_size = 768
└───────────────────────┘
        │
     [CLS] token embedding (768-dim)
        │
   ┌────┴──────────────────────────────┐
   │           │           │           │
   ▼           ▼           ▼           ▼
Sentiment   Behavior   Category    Urgency
Linear(768,3) Linear(768,8) Linear(768,12) Linear(768,4)
   │           │           │           │
   ▼           ▼           ▼           ▼
3 classes   8 classes   12 classes   4 classes

        Joint Loss = CE(sentiment) + CE(behavior) + CE(category) + CE(urgency)
```

---

## Tasks & Labels

### 1. Sentiment — 3 classes
| Label | Description |
|---|---|
| `negative` | Unhappy, frustrated, or complaining |
| `neutral` | Informational or mixed tone |
| `positive` | Satisfied or appreciative |

### 2. Behavior Score — 8 classes
| Label | Description |
|---|---|
| `very_poor` | Extremely negative interaction |
| `poor` | Below average tone |
| `below_average` | Slightly negative |
| `neutral` | Neither positive nor negative |
| `slightly_positive` | Mildly positive |
| `good` | Positive and constructive |
| `very_good` | Clearly positive |
| `excellent` | Highly positive and appreciative |

### 3. Issue Category — 12 classes
| Label | Description |
|---|---|
| `complaint` | General dissatisfaction |
| `technical_issue` | App or system bugs |
| `account_issue` | Login, access problems |
| `billing_payment` | Charges, refunds, payments |
| `service_delay` | Late delivery or slow response |
| `need_help` | Requests for assistance |
| `general_inquiry` | General questions |
| `policy_question` | Questions about rules/terms |
| `feature_request` | Suggestions for new features |
| `feedback` | General constructive feedback |
| `praise_appreciation` | Positive recognition |
| `customer_opinion` | Personal views or opinions |

### 4. Urgency — 4 classes
| Label | Description |
|---|---|
| `low` | No immediate action needed |
| `medium` | Should be addressed soon |
| `high` | Needs prompt attention |
| `critical` | Requires immediate escalation |

---

## Dataset

- **Source:** [Kaggle — MultiTask NLP Dataset](https://www.kaggle.com/datasets/viveksingh2400/multitask-nlp)
- **File:** `open_multitask_nlp_v1_train.csv`
- **Size:** ~20,180 samples
- **Columns:** `text`, `sentiment`, `behavior_score`, `category_1`, `urgency`
- **Duplicates:** 0
- **Class balance:** Near-uniform across all 12 categories (~1,580–1,850 samples each)

**Key data insight — sentiment × urgency correlation:**

| Sentiment | Critical | High | Low | Medium |
|---|---|---|---|---|
| negative | 2,528 | 4,214 | 0 | 0 |
| neutral | 0 | 828 | 4,100 | 6,660 |
| positive | 0 | 0 | 1,850 | 0 |

Sentiment and urgency are strongly correlated by design — negative messages are always high/critical, positive messages are always low urgency. This makes them easier to learn jointly.

---

## Model Design

### Why DistilBERT?
- 40% smaller and 60% faster than BERT-base with ~97% of BERT's performance on NLP benchmarks
- Sufficient capacity for short customer message classification (max 128 tokens)
- Efficient for deployment in inference-heavy production environments

### Why Multi-Task Learning?
- A single shared encoder captures general language understanding once
- All 4 tasks benefit from shared representations — sentiment understanding helps urgency prediction, category helps behavior scoring
- More efficient than training 4 separate models
- Single forward pass at inference time — lower latency

### Architecture Details

```python
class MultiTaskBert(torch.nn.Module):
    def __init__(self, model_name):
        self.encoder = AutoModel.from_pretrained(model_name)   # DistilBERT
        hidden = self.encoder.config.hidden_size               # 768

        # 4 independent task heads
        self.sentiment = nn.Linear(hidden, 3)
        self.behavior  = nn.Linear(hidden, 8)
        self.category  = nn.Linear(hidden, 12)
        self.urgency   = nn.Linear(hidden, 4)

    def forward(self, input_ids, attention_mask, ...):
        cls = encoder(input_ids, attention_mask).last_hidden_state[:, 0]
        # CLS token → all 4 heads simultaneously
```

**Joint loss:**
```
Total Loss = CE(sentiment) + CE(behavior) + CE(category) + CE(urgency)
```

All tasks contribute equally to the gradient update — no task weighting applied.

---

## Training

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 64 |
| Max token length | 250 (training) / 128 (inference) |
| Epochs | 10 |
| Train/Val split | 67% / 33% |
| Hardware | Google Colab T4 GPU |
| Loss function | CrossEntropyLoss (summed across 4 tasks) |

---

## Results

Validation accuracy after 10 epochs:

| Task | Accuracy | Notes |
|---|---|---|
| Sentiment | **100%** | Perfectly separable — strongly correlated with urgency |
| Category | **100%** | 12-class, well-balanced, cleanly learned |
| Urgency | **~80%** | Good performance, some overlap between `high` and `critical` |
| Behavior Score | **~50%** | Plateaued — 8-class ordinal task with high inter-class similarity |

**Note on behavior score:** The ~50% accuracy on behavior score is a known limitation of the dataset. The 8 ordinal classes (very_poor → excellent) have high semantic overlap and are difficult to distinguish even for human annotators. Sentiment and category converging to 100% first confirms the encoder is learning correctly — behavior score is a genuinely harder signal to extract from text alone.

---

## Project Structure

```
.
├── backend/
│   ├── backend.py              # FastAPI app — all endpoints
│   ├── predictsentiment.py     # MultiTaskBert model + MultitaskPredictor class
│   ├── multitask_model.pt      # Trained model weights (not in repo, download separately)
│   ├── multitask_tokenizer/    # Saved DistilBERT tokenizer
│   └── data/
│       └── feedbacks.jsonl     # Persisted feedback records
│
├── frontend/
│   ├── userfrontend.py         # Customer-facing feedback submission UI
│   └── adminfrontend.py        # Admin support dashboard with filters
│
├── notebooks/
│   ├── 00-dataoverview.ipynb   # EDA — class distributions, crosstabs, data quality
│   └── 01_setup_distbert_data_model.ipynb  # Full training pipeline
│
└── label_maps.json             # Label → index mappings for all 4 tasks
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip or uv

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd multitask-feedback-classifier
```

### 2. Install dependencies

```bash
pip install torch transformers fastapi uvicorn streamlit pydantic scikit-learn pandas
```

### 3. Download model weights

Place the trained model file and tokenizer in the `backend/` directory:

```
backend/multitask_model.pt
backend/multitask_tokenizer/
```

### 4. Start the backend

```bash
uvicorn backend.backend:app --reload --port 8000
```

### 5. Run the frontends

```bash
# Customer feedback form
streamlit run frontend/userfrontend.py

# Admin dashboard
streamlit run frontend/adminfrontend.py
```

---

## API Reference

All endpoints served on `http://localhost:8000`

---

### `POST /sentiments`
Submit a customer message and receive a 4-task prediction.

**Request body:**
```json
{
  "user_id": "U1234",
  "text": "I have been unable to access my account for 3 days and no one has responded."
}
```

**Response:**
```json
{
  "message": "Feedback sent successfully"
}
```

**Prediction stored in `feedbacks.jsonl`:**
```json
{
  "feedback_id": "FB-a3f9c812",
  "user_id": "U1234",
  "text": "I have been unable to access my account for 3 days...",
  "prediction": {
    "sentiment": "negative",
    "behavior": "very_poor",
    "category": "account_issue",
    "urgency": "critical"
  },
  "status": "pending",
  "created_at": "2025-06-07T10:30:00+00:00",
  "solved_at": null
}
```

---

### `GET /All`
Returns all feedback records regardless of status or category.

---

### `GET /Pending`
Returns all records with `status = "pending"`.

---

### `GET /Pending/Issues-Complaints`
Returns pending records in categories: `complaint`, `service_delay`, `account_issue`, `technical_issue`.

---

### `GET /Pending/payment-problems`
Returns pending records in category: `billing_payment`.

---

### `GET /Pending/help-information-requests`
Returns pending records in categories: `general_inquiry`, `need_help`, `policy_question`, `feature_request`.

---

### `GET /Pending/positive-opinion`
Returns pending records in categories: `praise_appreciation`, `customer_opinion`, `feedback`.

---

### `GET /All/{category-group}`
Same as above endpoints but returns all records regardless of status.

---

## Frontends

### Customer Frontend (`userfrontend.py`)
- Simple text form for submitting feedback
- Input validation: requires user ID, rejects empty text
- Strips emojis and normalizes whitespace before submission
- Sends cleaned text to `/sentiments` endpoint

### Admin Dashboard (`adminfrontend.py`)
- Filter by **status** (All / Pending) and **category group** (All / payment problems / positive opinions / help requests / issues & complaints)
- Displays results as an interactive Streamlit dataframe
- Handles 404 (no records) and connection errors gracefully
