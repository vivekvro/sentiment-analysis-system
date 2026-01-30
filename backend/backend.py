from backend.predictsentiment import predictor
from pydantic import BaseModel,field_validator
import re
from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import uuid
import json


app = FastAPI()
pred = predictor()
class TextRequest(BaseModel):
    user_id: str
    text: str
    @field_validator("text", mode="before")
    @classmethod
    def remove_newlines(cls,text):
        return re.sub(r"\s+", " ", text).strip()


# =========================
# API Endpoint
# =========================

FILE_PATH = "backend/data/feedbacks.jsonl"


@app.post("/sentiments")
def return_sentiment(req: TextRequest):
    prediction = pred.predict(req.text)

    record = {
        "feedback_id":f"FB-{uuid.uuid4().hex[:8]}",
        "user_id": req.user_id,
        "text": req.text,
        "prediction": prediction,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "solved_at": None
    }
    with open(FILE_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return JSONResponse(status_code=200,content={"message": "Feedback sent successfully"})


def load_data():
    data = []
    with open(FILE_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data



@app.get("/Pending")
def return_pending_supports():
    records =  [d for d in load_data() if d["status"] == "pending"]
    if not records:
        raise HTTPException(404, "No pending records found")
    return records


@app.get("/All")
def Records():
    records =  [d for d in load_data()]
    if not records:
        raise HTTPException(404, "No records found")
    return records

#------------------------------------------------------------------------------

@app.get("/Pending/payment-problems")
def payment_problems():
    records = [
        d for d in load_data()
        if d["prediction"]["category"] == "billing_payment"
        and d["status"] == "pending"
    ]

    if not records:
        raise HTTPException(404, "No pending Payment Problems found")

    return records


@app.get("/All/payment-problems")
def payment_problems():
    records = [
        d for d in load_data()
        if d["prediction"]["category"] == "billing_payment"
    ]

    if not records:
        raise HTTPException(404, "No pending Payment Problems found")

    return records
#------------------------------------------------------------------------------
@app.get("/Pending/positive-opinion")
def positive_opinion():
    records = [
        d for d in load_data()
        if d["prediction"]["category"] in [
            "praise_appreciation",
            "customer_opinion",
            "feedback"
        ]
        and d["status"] == "pending"
    ]

    if not records:
        raise HTTPException(404, "No pending Positive Opinion records found")

    return records

@app.get("/All/positive-opinion")
def positive_opinion():
    records = [
        d for d in load_data()
        if d["prediction"]["category"] in [
            "praise_appreciation",
            "customer_opinion",
            "feedback"
        ]
    ]

    if not records:
        raise HTTPException(404, "No pending Positive Opinion records found")

    return records

#------------------------------------------------------------------------------
@app.get("/Pending/help-information-requests")
def PHIR():
    records = [
        d for d in load_data()
            if d["prediction"]['category'] in [
                "general_inquiry",
                "need_help",
                "policy_question",
                "feature_request"]
                and d["status"] == "pending"]
    if not records:
        raise HTTPException(
                    status_code=404,
                    detail="No pending help or information requests found"
                )
    return records
@app.get("/All/help-information-requests")
def AHIR():
    records = [
        d for d in load_data()
            if d["prediction"]['category'] in [
                "general_inquiry",
                "need_help",
                "policy_question",
                "feature_request"]]
    if not records:
        raise HTTPException(
                    status_code=404,
                    detail="No help or information requests found"
                )
    return records

#------------------------------------------------------------------------------
@app.get("/Pending/Issues-Complaints")
def IC():
    records = [d for d in load_data()
            if d["prediction"]["category"] in [
                "complaint",
                "service_delay",
                "account_issue",
                "technical_issue"]
                and d["status"] == "pending"]
    if not records:
        raise HTTPException(
            status_code=404,
            detail="No pending Issues-Complaints found"
        )
    return records

@app.get("/All/Issues-Complaints")
def IC():
    records = [d for d in load_data()
            if d["prediction"]["category"] in [
                "complaint",
                "service_delay",
                "account_issue",
                "technical_issue"]]
    if not records:
        raise HTTPException(
            status_code=404,
            detail="No Issues-Complaints found"
        )
    return records

#------------------------------------------------------------------------------
@app.get("/Records/{user_id}")
def Records(user_id:str):
    return [d for d in load_data if d['user_id'==user_id]]


