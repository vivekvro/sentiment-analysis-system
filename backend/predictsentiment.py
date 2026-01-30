# =========================
# Imports
# =========================
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import json
from pathlib import Path
# =========================
# Multitask Model
# =========================
class MultiTaskBert(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.sentiment = torch.nn.Linear(hidden, 3)
        self.behavior  = torch.nn.Linear(hidden, 8)
        self.category  = torch.nn.Linear(hidden, 12)
        self.urgency   = torch.nn.Linear(hidden, 4)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        sentiment_labels=None,
        behavior_labels=None,
        category_labels=None,
        urgency_labels=None
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]

        s_logits = self.sentiment(cls)
        b_logits = self.behavior(cls)
        c_logits = self.category(cls)
        u_logits = self.urgency(cls)

        # ✅ Always return logits
        output = {
            "sentiment_logits": s_logits,
            "behavior_logits": b_logits,
            "category_logits": c_logits,
            "urgency_logits": u_logits
        }

        # Optional loss (used only during training)
        if sentiment_labels is not None:
            loss = (
                self.loss_fn(s_logits, sentiment_labels) +
                self.loss_fn(b_logits, behavior_labels) +
                self.loss_fn(c_logits, category_labels) +
                self.loss_fn(u_logits, urgency_labels)
            )
            output["loss"] = loss

        return output

# =========================
# Label maps (USED FOR DECODING)
# =========================
label_maps = {
    "sentiment": {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    },
    "behavior": {
        "very_poor": 0,
        "poor": 1,
        "below_average": 2,
        "neutral": 3,
        "slightly_positive": 4,
        "good": 5,
        "very_good": 6,
        "excellent": 7
    },
    "category": {
        "account_issue": 0,
        "complaint": 1,
        "feedback": 2,
        "need_help": 3,
        "technical_issue": 4,
        "feature_request": 5,
        "billing_payment": 6,
        "policy_question": 7,
        "service_delay": 8,
        "general_inquiry": 9,
        "praise_appreciation": 10,
        "customer_opinion": 11
    },
    "urgency": {
        "low": 0,
        "medium": 1,
        "high": 2,
        "critical": 3
    }
}

# 🔧 FIX: consistent filename
with open("label_maps.json", "w") as f:
    json.dump(label_maps, f, indent=2)

# =========================
# Predictor Class
# =========================
class MultitaskPredictor:
    def __init__(
        self,
        model_class,
        model_name,
        model_path,
        tokenizer_path,
        label_map_path,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # label maps
        with open(label_map_path) as f:
            label_maps = json.load(f)

        self.inv_maps = {
            k: {v: k for k, v in label_maps[k].items()}
            for k in label_maps
        }

        # 🔧 FIX: correct model loading order
        self.model = model_class(model_name)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, max_length=128):  # 🔧 FIX: realistic max_length
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        return {
            "sentiment": self.inv_maps["sentiment"][
                torch.argmax(outputs["sentiment_logits"], dim=1).item()
            ],
            "behavior": self.inv_maps["behavior"][
                torch.argmax(outputs["behavior_logits"], dim=1).item()
            ],
            "category": self.inv_maps["category"][
                torch.argmax(outputs["category_logits"], dim=1).item()
            ],
            "urgency": self.inv_maps["urgency"][
                torch.argmax(outputs["urgency_logits"], dim=1).item()
            ]
        }

# =========================
# FastAPI App
# =========================


# 🔧 FIX: load model ONCE at startup
def predictor():
    BASE_DIR = Path(__file__).resolve().parent
    return MultitaskPredictor(
    model_class=MultiTaskBert,
    model_name="distilbert-base-uncased",   # ✅ correct model name
    model_path=BASE_DIR/"multitask_model.pt",
    tokenizer_path=BASE_DIR/"multitask_tokenizer",
    label_map_path="label_maps.json"
)



