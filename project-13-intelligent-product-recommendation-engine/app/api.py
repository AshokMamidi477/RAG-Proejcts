"""api.py — FastAPI Product Recommendation Engine"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.recommender import recommend


# -----------------------------
# Logging
# -----------------------------

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(
    title="Intelligent Product Recommendation Engine",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



# -----------------------------
# Request Models
# -----------------------------

class RecommendRequest(BaseModel):

    query: str

    context: str = ""

    category: Optional[str] = None



# -----------------------------
# Response Models
# -----------------------------

class RecommendationResponse(BaseModel):

    rank: Optional[int] = None

    product_id: str

    name: str

    reason: Optional[str] = None

    price: Optional[float] = None

    category: Optional[str] = None

    rating: Optional[float] = None



class RecommendResponse(BaseModel):

    recommendations: list[RecommendationResponse]

    count: int



# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "Product Recommendation Engine"
    }



# -----------------------------
# Recommendation API
# -----------------------------

@app.post(
    "/recommend",
    response_model=RecommendResponse
)
def get_recommendations(
    req: RecommendRequest
):

    try:

        logger.info(
            f"Recommendation request: {req.query}"
        )


        results = recommend(
            query=req.query,
            context=req.context,
            category=req.category

        )

        return {

            "recommendations": results,

            "count": len(results)

        }


    except Exception as e:

        logger.exception(
            "Recommendation failed"
        )

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )