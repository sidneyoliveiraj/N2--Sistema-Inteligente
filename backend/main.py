import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from layer1_nlp import SentimentClassifier
from layer2_fuzzy import FuzzyEngagementSystem
from layer3_sa import StudySchedulerSA

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# instâncias globais dos três modelos
nlp: SentimentClassifier | None = None
fuzzy: FuzzyEngagementSystem | None = None
sa: StudySchedulerSA | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp, fuzzy, sa
    log.info("Carregando modelos...")
    nlp   = SentimentClassifier()
    fuzzy = FuzzyEngagementSystem()
    sa    = StudySchedulerSA()
    log.info("Todos os modelos prontos.")
    yield


app = FastAPI(title="N2 – Sistema Inteligente", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# schemas de entrada
class FeedbackIn(BaseModel):
    text:   str
    rating: float = Field(..., ge=1, le=5)


class TopicIn(BaseModel):
    name:      str
    feedbacks: List[FeedbackIn]


class PipelineRequest(BaseModel):
    topics: List[TopicIn]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "nlp":   nlp   is not None,
            "fuzzy": fuzzy is not None,
            "sa":    sa    is not None,
        },
        "nlp_accuracy": round(nlp.accuracy, 4) if nlp else None,
    }


@app.post("/analyze")
def analyze(body: FeedbackIn):
    if not nlp or not fuzzy:
        raise HTTPException(503, "Modelos ainda não carregados")

    layer1 = nlp.predict(body.text)
    layer2 = fuzzy.compute(layer1["positive_prob"], body.rating)
    return {"layer1": layer1, "layer2": layer2}


@app.post("/pipeline")
def pipeline(body: PipelineRequest):
    if not nlp or not fuzzy or not sa:
        raise HTTPException(503, "Modelos ainda não carregados")

    topics_result = []

    for topic in body.topics:
        feedbacks_analyzed = []
        total_pos_prob = 0.0
        total_rating   = 0.0

        for fb in topic.feedbacks:
            layer1 = nlp.predict(fb.text)
            feedbacks_analyzed.append({
                "text":   fb.text,
                "rating": fb.rating,
                "layer1": layer1,
            })
            total_pos_prob += layer1["positive_prob"]
            total_rating   += fb.rating

        n = len(topic.feedbacks)
        avg_pos_prob = total_pos_prob / n
        avg_rating   = total_rating   / n

        layer2 = fuzzy.compute(avg_pos_prob, avg_rating)

        topics_result.append({
            "name":             topic.name,
            "feedbacks":        feedbacks_analyzed,
            "avg_pos_prob":     round(avg_pos_prob, 3),
            "avg_rating":       round(avg_rating,   2),
            "layer2":           layer2,
            "engagement_score": layer2["engagement_score"],
        })

    # passa os scores para o SA ordenar a sequência de estudo
    sa_input  = [{"name": t["name"], "engagement_score": t["engagement_score"]} for t in topics_result]
    sa_result = sa.optimize(sa_input)

    return {"topics": topics_result, "layer3": sa_result}
