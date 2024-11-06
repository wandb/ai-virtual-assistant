# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The definition of the Llama Index chain server."""
import os
import json
import logging
from typing import List
import bleach
import importlib
import random

# For session response
from fastapi import Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel, Field, validator

from src.analytics.datastore.session_manager import SessionManager

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "Health",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {"name": "Feedback", "description": "APIs for storing useful information for data flywheel."},
    {"name": "Session", "description": "APIs for fetching useful information for different sessions."},
    {"name": "User Data", "description": "APIs for fetching user specific information."},
]

# create the FastAPI server
app = FastAPI(title="Analytics API's for AI Virtual Assistant for Customer Service",
    description="This API schema describes all the analytics endpoints exposed for the AI Virtual Assistant for Customer Service NIM Blueprint",
    version="1.0.0",
        docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
)

EXAMPLE_DIR = "./"
# List of fallback responses sent out for any Exceptions from /generate endpoint
FALLBACK_RESPONSES = [
    "Please try re-phrasing, I am likely having some trouble with that conversation.",
    "I will get better with time, please retry with a different conversation.",
    "I wasn't able to process your conversation. Let's try something else.",
    "Something went wrong. Could you try again in a few seconds with a different conversation.",
    "Oops, that proved a tad difficult for me, can you retry with another conversation?"
]

# Allow access in browser from RAG UI and Storybook (development)
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

# For session response
# Updated model for session response
class SessionInfo(BaseModel):
    session_id: str = Field(..., description="The ID of the session")
    # Make the start_time mandatory
    # TODO Make start time mandatory when needed
    start_time: Optional[datetime] = Field(None, description="The start time of the session")
    end_time: Optional[datetime] = Field(None, description="The end time of the session")
    # Add other session-related if needed

    # TODO uncomment while actual implementation
    # @validator('start_time', 'end_time')
    # def ensure_utc(cls, v):
    #     if v is not None and v.tzinfo is None:
    #         raise ValueError("Datetime must be in UTC format")
    #     return v

class SessionsResponse(BaseModel):
    session_id: str = Field(..., description="The ID of the session")
    user_id: Optional[str] = Field(..., description="The ID of the user")
    start_time: Optional[datetime] = Field(None, description="The start time of the session")
    end_time: Optional[datetime] = Field(None, description="The end time of the session")

class SessionSummaryResponse(BaseModel):
    session_info: SessionInfo
    summary: str = Field(..., description="The generated summary of the session")
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(..., description="The sentiment of the text, which can be positive, negative, or neutral.")


class SessionConversationMessage(BaseModel):
    """Definition of the Chat Message type."""
    role: str = Field(description="Role for a message AI, User and System", default="user", max_length=256, pattern=r'[\s\S]*')
    content: str = Field(description="The input query/prompt to the pipeline.", default="I am going to Paris, what should I see?", max_length=131072, pattern=r'[\s\S]*')
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(..., description="The sentiment of the text, which can be positive, negative, or neutral.")

    @validator('role')
    def validate_role(cls, value):
        """ Field validator function to validate values of the field role"""
        value = bleach.clean(value, strip=True)
        valid_roles = {'user', 'assistant', 'system'}
        if value.lower() not in valid_roles:
            raise ValueError("Role must be one of 'user', 'assistant', or 'system'")
        return value.lower()

    @validator('content')
    def sanitize_content(cls, v):
        """ Feild validator function to santize user populated feilds from HTML"""
        return bleach.clean(v, strip=True)

class SessionConversationResponse(BaseModel):
    session_info: SessionInfo
    messages: List[SessionConversationMessage] = Field(..., description="The list of messages in the conversation")

class FeedbackRequest(BaseModel):
    """Definition of the Feedback Request data type."""
    feedback: float = Field(..., description="A unique identifier representing your end-user.", ge=-1.0, le=1.0)
    session_id: str = Field(..., description="A unique identifier representing the session associated with the response.")

class FeedbackResponse(BaseModel):
    """Definition of the Feedback Request data type."""
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

class PurchaseInfo(BaseModel):
    customer_id: str
    order_id: str
    product_name: str
    order_date: str
    quantity: Optional[int]
    order_amount: Optional[float]
    order_status: Optional[str]
    return_status: Optional[str]
    return_start_date: Optional[str]
    return_received_date: Optional[str]
    return_completed_date: Optional[str]
    return_reason: Optional[str]
    notes: Optional[str]

@app.on_event("startup")
def import_example() -> None:
    """
    Import the example class from the specified example file.

    """
    file_location = os.path.join(EXAMPLE_DIR, os.environ.get("EXAMPLE_PATH", "basic_rag/llamaindex"))

    for root, dirs, files in os.walk(file_location):
        for file in files:
            if file == "main.py":
                # Import the specified file dynamically
                spec = importlib.util.spec_from_file_location(name="main", location=os.path.join(root, file))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                logger.info(f"Found analytics api {file}")
                # Get the Analytics app
                app.analytics = module
                # break  # Stop the loop once we find and load main.py

    app.session_manager = SessionManager()

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": jsonable_encoder(exc.errors(), exclude={"input"})})

@app.get("/health", tags=["Health"], response_model=HealthResponse, responses={
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def health_check():
    """
    Perform a Health Check

    Returns 200 when service is up. This does not check the health of downstream services.
    """

    response_message = "Service is up."
    return HealthResponse(message=response_message)

@app.get("/sessions", tags=["Session"], response_model=List[SessionsResponse], responses={
    404: {
        "description": "No Sessions Found",
        "content": {
            "application/json": {
                "example": {"detail": "No sessions found for the specified time range"}
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
# TODO Add the instrumentation_wrapper when required
#@llamaindex_instrumentation_wrapper
async def get_sessions(
    hours: int = Query(..., description="Last K hours, for which sessions info is extracted"),
) -> List[SessionsResponse]:
    """
    Retrieve session information in last k hours
    """
    try:
        result = app.session_manager.get_conversations_in_last_h_hours(hours)

        resp = []
        for r in result:
            resp.append(
                SessionsResponse(
                    session_id=r.get("session_id"),
                    user_id=r.get("user_id"),
                    start_time=r.get("start_conversation_time"),
                    end_time=r.get("last_conversation_time"),
                )
            )

        return resp

    except Exception as e:
        logger.error(f"Error in GET /sessions endpoint. Error details: {e}")
        return JSONResponse(content={"detail": "Error occurred while retrieving session information"}, status_code=500)


@app.get("/session/summary", tags=["Session"], response_model=SessionSummaryResponse, responses={
    404: {
        "description": "Session Not Found",
        "content": {
            "application/json": {
                "example": {"detail": "Session not found"}
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def generate_session_summary(
    request: Request,
    session_id: str = Query(..., description="The ID of the session")
) -> SessionSummaryResponse:
    """Generate a summary and sentiment analysis for the specified session."""

    try:
        result = app.analytics.generate_session_summary(session_id)
        summary = result.get("summary")
        sentiment = result.get("sentiment")
        session_info = SessionInfo(session_id=session_id, start_time=result.get("start_time"), end_time=result.get("end_time"))
        response = SessionSummaryResponse(session_info=session_info, summary=summary, sentiment=sentiment)
        return response

    except ValueError as e:
        logger.error(f"Session not found for ID {session_id}. Error details: {e}")
        return SessionSummaryResponse(session_info=SessionInfo(session_id=session_id), summary=random.choice(FALLBACK_RESPONSES), sentiment="neutral")

    except Exception as e:
        logger.error(f"Error in GET /session/summary endpoint. Error details: {e}")
        return SessionSummaryResponse(session_info=SessionInfo(session_id=session_id), summary=random.choice(FALLBACK_RESPONSES), sentiment="neutral")


@app.get("/session/conversation", tags=["Session"], response_model=SessionConversationResponse, responses={
    404: {
        "description": "Session Not Found",
        "content": {
            "application/json": {
                "example": {"detail": "Session not found"}
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def get_session_conversation(
    request: Request,
    session_id: str = Query(..., description="The ID of the session"),
) -> SessionConversationResponse:
    """Retrieve the conversation and sentiment for the specified session."""

    try:
        result = app.analytics.generate_sentiment_for_query(session_id)

        message = []
        for msg in result.get("messages", []):
            message.append(SessionConversationMessage(role=msg.get("role"), content=msg.get("content"), sentiment=msg.get("sentiment")))

        session_info = SessionInfo(session_id=session_id, start_time=result.get("session_info", {}).get("start_time"), end_time=result.get("session_info", {}).get("start_time"))
        response = SessionConversationResponse(session_info=session_info, messages=message)
        return response

    except ValueError as e:
        logger.error(f"Session not found for ID {session_id}. Error details: {e}")
        raise HTTPException(status_code=404, detail="Session not found. Please check the session ID or end the session.")

    except Exception as e:
        logger.error(f"Error in GET /session/conversation endpoint. Error details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")


@app.post("/get_user_purchase_history", tags=["User Data"], response_model=List[PurchaseInfo], responses={
    404: {
        "description": "Session Not Found",
        "content": {
            "application/json": {
                "example": {"detail": "Session not found"}
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def get_user_purchase_history(
    user_id: str,
) -> List[PurchaseInfo]:
    """Get purchase history for user"""
    try:

        logger.info(f"Fetching purchase history for user {user_id}")
        product_info = app.session_manager.get_purchase_history(user_id)
        response = []
        for product in product_info:
            response.append(PurchaseInfo(
                customer_id=str(product.get("customer_id")),
                order_id=str(product.get("order_id")),
                product_name=str(product.get("product_name")),
                order_date=str(product.get("order_date")),
                quantity=str(product.get("quantity")),
                order_amount=str(product.get("order_amount")),
                order_status=str(product.get("order_status")),
                return_status=str(product.get("return_status")),
                return_start_date=str(product.get("return_start_date")),
                return_received_date=str(product.get("return_received_date")),
                return_completed_date=str(product.get("return_completed_date")),
                return_reason=str(product.get("return_reason")),
                notes=str(product.get("notes")),
                )
            )
        return response
    except Exception as e:
        logger.error(f"Error in GET /get_user_purchase_history endpoint. Error details: {e}")
        return []

@app.post("/feedback/sentiment", tags=["Feedback"], response_model=FeedbackResponse, responses={
    404: {
        "description": "Session Not Found",
        "content": {
            "application/json": {
                "example": {"detail": "Session not found"}
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def store_sentiment_feedback(
    request: Request,
    feedback: FeedbackRequest,
) -> FeedbackResponse:
    """Store user feedback for the sentiment analysis of a conversation session."""
    # TODO: Add validation to store feeedback only when conversation exist

    try:
        logger.info("Storing user feedback for sentiment")
        app.session_manager.save_sentiment_feedback(feedback.session_id, feedback.feedback)
        return FeedbackResponse(message="Sentiment feedback saved successfully")
    except Exception as e:
        logger.error(f"Error in GET /feedback/sentiment endpoint. Error details: {e}")
        return FeedbackResponse(message="Failed to store sentiment feedback")

@app.post("/feedback/summary", tags=["Feedback"], response_model=FeedbackResponse, responses={
    404: {
        "description": "Session Not Found",
        "content": {
            "application/json": {
                "example": {"detail": "Session not found"}
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def store_summary_feedback(
    request: Request,
    feedback: FeedbackRequest,
) -> FeedbackResponse:
    """Store user feedback for the summary of a conversation session."""
    # TODO: Add validation to store feeedback only when conversation exist
    try:
        logger.info("Storing user feedback for summary")
        app.session_manager.save_summary_feedback(feedback.session_id, feedback.feedback)
        return FeedbackResponse(message="Summary feedback saved successfully")
    except Exception as e:
        logger.error(f"Error in GET /feedback/summary endpoint. Error details: {e}")
        return FeedbackResponse(message="Failed to store summary feedback")


@app.post("/feedback/session", tags=["Feedback"], response_model=FeedbackResponse, responses={
    404: {
        "description": "Session Not Found",
        "content": {
            "application/json": {
                "example": {"detail": "Session not found"}
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def store_conversation_session_feedback(
    request: Request,
    feedback: FeedbackRequest,
) -> FeedbackResponse:
    """Store user feedback for the overall conversation session."""
    try:
        # TODO: Add validation to store feeedback only when conversation exist
        logger.info("Storing user feedback for summary")
        app.session_manager.save_session_feedback(feedback.session_id, feedback.feedback)
        return FeedbackResponse(message="Session feedback saved successfully")
    except Exception as e:
        logger.error(f"Error in GET /feedback/session endpoint. Error details: {e}")
        return FeedbackResponse(message="Failed to store Session feedback")
