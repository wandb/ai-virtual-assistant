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
from uuid import uuid4
import logging
from typing import List
import importlib
import bleach
import time
import prometheus_client
import asyncio
import random
import weave
import re
from traceback import print_exc

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel, Field, validator, constr
from src.agent.cache.session_manager import SessionManager
from src.agent.datastore.datastore import Datastore
from src.agent.utils import remove_state_from_checkpointer

from langgraph.errors import GraphRecursionError
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

if "WANDB_API_KEY" in os.environ:
    if "WANDB_PROJECT" in os.environ:
        client = weave.init(os.environ.get("WANDB_PROJECT"))
    else:
        logger.warning("No WANDB_PROJECT env var set, using default W&B project: nv-ai-virtual-assistant")
        client = weave.init("nv-ai-virtual-assistant")
else:
    logger.warning("No WANDB_API_KEY env var set, No logging to W&B")

tags_metadata = [
    {
        "name": "Health",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {"name": "Feedback", "description": "APIs for storing useful information for data flywheel."},
    {"name": "Session Management", "description": "APIs for managing sessions."},
    {"name": "Inference", "description": "Core APIs for interacting with the agent."},
]

# create the FastAPI server
app = FastAPI(title="Agent API's for AI Virtual Assistant for Customer Service",
    description="This API schema describes all the core agentic endpoints exposed by the AI Virtual Assistant for Customer Service NIM Blueprint",
    version="1.0.0",
        docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
)

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

EXAMPLE_DIR = "./"

# List of fallback responses sent out for any Exceptions from /generate endpoint
FALLBACK_RESPONSES = [
    "Please try re-phrasing, I am likely having some trouble with that question.",
    "I will get better with time, please try with a different question.",
    "I wasn't able to process your input. Let's try something else.",
    "Something went wrong. Could you try again in a few seconds with a different question?",
    "Oops, that proved a tad difficult for me, can you retry with another question?"
]

class Message(BaseModel):
    """Definition of the Chat Message type."""
    role: str = Field(description="Role for a message AI, User and System", default="user", max_length=256, pattern=r'[\s\S]*')
    content: str = Field(description="The input query/prompt to the pipeline.", default="Hello what can you do?", max_length=131072, pattern=r'[\s\S]*')

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
        """ Field validator function to santize user populated feilds from HTML"""
        v = bleach.clean(v, strip=True)
        if not v:  # Check for empty string
            raise ValueError("Message content cannot be empty.")
        return v

class Prompt(BaseModel):
    """Definition of the Prompt API data type."""
    messages: List[Message] = Field(..., description="A list of messages comprising the conversation so far. The roles of the messages must be alternating between user and assistant. The last input message should have role user. A message with the the system role is optional, and must be the very first message if it is present.", max_items=50000)
    user_id: str = Field(None, description="A unique identifier representing your end-user.")
    session_id: str = Field(..., description="A unique identifier representing the session associated with the response.")

class ChainResponseChoices(BaseModel):
    """ Definition of Chain response choices"""
    index: int = Field(default=0, ge=0, le=256, format="int64")
    message: Message = Field(default=Message())
    finish_reason: str = Field(default="", max_length=4096, pattern=r'[\s\S]*')

class ChainResponse(BaseModel):
    """Definition of Chain APIs resopnse data type"""
    id: str = Field(default="", max_length=100000, pattern=r'[\s\S]*')
    choices: List[ChainResponseChoices] = Field(default=[], max_items=256)
    session_id: str = Field(None, description="A unique identifier representing the session associated with the response.")

class DocumentSearch(BaseModel):
    """Definition of the DocumentSearch API data type."""

    query: str = Field(description="The content or keywords to search for within documents.", max_length=131072, pattern=r'[\s\S]*', default="")
    top_k: int = Field(description="The maximum number of documents to return in the response.", default=4, ge=0, le=25, format="int64")

class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""
    content: str = Field(description="The content of the document chunk.", max_length=131072, pattern=r'[\s\S]*', default="")
    filename: str = Field(description="The name of the file the chunk belongs to.", max_length=4096, pattern=r'[\s\S]*', default="")
    score: float = Field(..., description="The relevance score of the chunk.")

class DocumentSearchResponse(BaseModel):
    """Represents a response from a document search."""
    chunks: List[DocumentChunk] = Field(..., description="List of document chunks.", max_items=256)

class DocumentsResponse(BaseModel):
    """Represents the response containing a list of documents."""
    documents: List[constr(max_length=131072, pattern=r'[\s\S]*')] = Field(description="List of filenames.", max_items=1000000, default=[])

class HealthResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

class CreateSessionResponse(BaseModel):
    session_id: str = Field(max_length=4096)

class EndSessionResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

class DeleteSessionResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

class FeedbackRequest(BaseModel):
    """Definition of the Feedback Request data type."""
    feedback: float = Field(..., description="A unique identifier representing your end-user.", ge=-1.0, le=1.0)
    session_id: str = Field(..., description="A unique identifier representing the session associated with the response.")

class FeedbackResponse(BaseModel):
    """Definition of the Feedback Request data type."""
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

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

                # Get the Agent app
                app.agent = module
                break  # Stop the loop once we find and load agent.py

    # Initialize session manager during startup
    app.session_manager = SessionManager()

    # Initialize database to store conversation permanently
    app.database = Datastore()

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


@app.get("/metrics", tags=["Health"])
async def get_metrics():
    return Response(content=prometheus_client.generate_latest(), media_type="text/plain")


@app.get("/create_session", tags=["Session Management"], response_model=CreateSessionResponse, responses={
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def create_session():

    # Try for fix number of time, if no unique session_id is found raise Error
    for _ in range(5):
        session_id = str(uuid4())

        # Ensure session_id created does not exist in cache
        if not app.session_manager.is_session(session_id):
            # Ensure session_id created does not exist in datastore (permanenet store like postgres)
            if not app.database.is_session(session_id):
                # Create a session on cache for validation
                app.session_manager.create_session(session_id)
                return CreateSessionResponse(session_id=session_id)

    raise HTTPException(status_code=500, detail="Unable to generate session_id")


@app.get("/end_session", tags=["Session Management"], response_model=EndSessionResponse, responses={
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def end_session(session_id):
    logger.info(f"Fetching conversation for {session_id} from cache")
    session_info = app.session_manager.get_session_info(session_id)
    logger.info(f"Session INFO: {session_info}")
    if not session_info or not session_info.get("start_conversation_time", None):
        logger.info("No conversation found in session")
        return EndSessionResponse(message="Session not found. Create session before trying out")

    if session_info.get("last_conversation_time"):
        # If there is no conversation history then don't port it to datastore
        logger.info(f"Storing conversation for {session_id} in database")
        app.database.store_conversation(session_id, session_info.get("user_id"), session_info.get("conversation_hist"), session_info.get("last_conversation_time"), session_info.get("start_conversation_time"))

    # Once the conversation is ended and ported to permanent storage, clear cache with given session_id
    logger.info(f"Deleting conversation for {session_id} from cache")
    app.session_manager.delete_conversation(session_id)

    return EndSessionResponse(message="Session ended")


@app.delete("/delete_session", tags=["Session Management"], response_model=DeleteSessionResponse, responses={
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def delete_session(session_id):
    logger.info(f"Deleting conversation for {session_id}")
    session_info = app.session_manager.get_session_info(session_id)
    if not session_info:
        logger.info("No conversation found in session")
        return DeleteSessionResponse(message="Session info not found")

    logger.info(f"Deleting conversation for {session_id} from cache")
    app.session_manager.delete_conversation(session_id)

    logger.info(f"Deleting conversation for {session_id} in database")
    app.database.delete_conversation(session_id)

    logger.info(f"Deleting checkpointer for {session_id}")
    remove_state_from_checkpointer(session_id)
    return EndSessionResponse(message="Session info deleted")

@weave.op()
def fallback_response_generator(sentence: str, session_id: str = ""):
    """Mock response generator to simulate streaming predefined fallback responses."""

    # Simulate breaking the sentence into chunks (e.g., by word)
    sentence_chunks = sentence.split()  # Split the sentence by words
    resp_id = str(uuid4())  # unique response id for every query
    # Send each chunk (word) in the response
    for chunk in sentence_chunks:
        chain_response = ChainResponse(session_id=session_id, sentiment="")
        response_choice = ChainResponseChoices(
            index=0,
            message=Message(role="assistant", content=f"{chunk} ")
        )
        chain_response.id = resp_id
        chain_response.choices.append(response_choice)
        yield "data: " + str(chain_response.json()) + "\n\n"

    # End with [DONE] response
    chain_response = ChainResponse(session_id=session_id, sentiment="")
    response_choice = ChainResponseChoices(message=Message(role="assistant", content=" "), finish_reason="[DONE]")
    chain_response.id = resp_id
    chain_response.choices.append(response_choice)
    yield "data: " + str(chain_response.json()) + "\n\n"


@app.post(
    "/generate",
    tags=["Inference"],
    response_model=ChainResponse,
    responses={
        500: {
            "description": "Internal Server Error",
            "content": {"application/json": {"example": {"detail": "Internal server error occurred"}}},
        }
    },
)
@weave.op()
async def generate_answer(request: Request,
                          prompt: Prompt) -> StreamingResponse:
    """Generate and stream the response to the provided prompt."""

    logger.info(f"Input at /generate endpoint of Agent: {prompt.dict()}")

    try:
        user_query_timestamp = time.time()

        # Handle invalid session id
        if not app.session_manager.is_session(prompt.session_id):
            logger.error(f"No session_id created {prompt.session_id}. Please create session id before generate request.")
            print_exc()
            return StreamingResponse(fallback_response_generator(sentence=random.choice(FALLBACK_RESPONSES), session_id=prompt.session_id), media_type="text/event-stream")

        chat_history = prompt.messages
        # The last user message will be the query for the rag or llm chain
        last_user_message = next((message.content for message in reversed(chat_history) if message.role == 'user'), None)

        # Normalize the last user input and remove non-ascii characters
        last_user_message = re.sub(r'[^\x00-\x7F]+', '', last_user_message) # Remove all non-ascii characters
        last_user_message = re.sub(r'[\u2122\u00AE]', '', last_user_message) # Remove standard trademark and copyright symbols
        last_user_message = last_user_message.replace("~", "-")
        logger.info(f"Normalized user input: {last_user_message}")

        # Keep copy of unmodified query to store in db
        user_query = last_user_message

        log_level=os.environ.get('LOGLEVEL', 'INFO').upper()
        debug_langgraph = False
        if log_level == "DEBUG":
            debug_langgraph = True

        recursion_limit = int(os.environ.get('GRAPH_RECURSION_LIMIT', '6'))

        async def response_generator():

            try:
                resp_id = str(uuid4())
                is_exception = False
                # Variable to track if this is the first yield
                is_first_yield = True
                resp_str = ""
                last_content = ""

                logger.info(f"Chat History:  {app.session_manager.get_conversation(prompt.session_id)}")
                config = {"recursion_limit": recursion_limit,
                    "configurable": {"thread_id": prompt.session_id, "chat_history": app.session_manager.get_conversation(prompt.session_id)}}
                # Check for the interrupt
                snapshot = await app.agent.graph.aget_state(config)
                if not snapshot.next:
                    input_for_graph = {"messages":[("human", last_user_message)], "user_id": prompt.user_id}
                else:
                    if last_user_message.strip().startswith(("Yes", "yes", "Y", "y")):
                        # Just continue
                        input_for_graph = None
                    else:
                        last_item = snapshot.values.get("messages")[-1]
                        if last_item and hasattr(last_item, "tool_calls") and last_item.tool_calls:
                            input_for_graph = {
                                        "messages": [
                                            ToolMessage(
                                                tool_call_id=last_item.tool_calls[0]["id"],
                                                content=f"API call denied by user. Reasoning: '{last_user_message}'. Continue assisting, accounting for the user's input.",
                                            )
                                        ]
                                    }
                        elif not hasattr(last_item, "tool_calls"):
                            input_for_graph = {"messages":[("human", last_user_message)], "user_id": prompt.user_id}
                        else:
                            input_for_graph = None
                try:
                    function_start_time = time.time()
                    # Set Maximum time to wait for a step to complete, in seconds. Defaults to None
                    graph_timeout_env =  os.environ.get('GRAPH_TIMEOUT_IN_SEC', None)
                    app.agent.graph.step_timeout = int(graph_timeout_env) if graph_timeout_env else None
                    async for event in app.agent.graph.astream_events(input_for_graph, version="v2", config=config, debug=debug_langgraph):
                        kind = event["event"]
                        tags = event.get("tags", [])
                        if kind == "on_chain_end" and event['data'].get('output', "") == '__end__':
                            end_msgs = event['data']['input']['messages']
                            last_content = end_msgs[-1].content
                        if kind == "on_chat_model_stream" and "should_stream" in tags:
                            content = event["data"]["chunk"].content
                            resp_str += content
                            if content:
                                chain_response = ChainResponse()
                                response_choice = ChainResponseChoices(
                                    index=0,
                                    message=Message(
                                        role="assistant",
                                        content=content
                                    )
                                )
                                chain_response.id = resp_id
                                chain_response.session_id = prompt.session_id
                                chain_response.choices.append(response_choice)
                                logger.debug(response_choice)
                                # Check if this is the first yield
                                if is_first_yield:
                                    logger.info(f"Execution time until first yield:  {time.time() - function_start_time}")
                                    is_first_yield = False
                                yield "data: " + str(chain_response.json()) + "\n\n"

                    # If resp_str is empty after the loop, use the last AI message content
                    # If there is no Streaming response
                    if not resp_str and last_content:
                        chain_response = ChainResponse()
                        response_choice = ChainResponseChoices(
                            index=0,
                            message=Message(
                                role="assistant",
                                content=last_content
                            )
                        )
                        chain_response.id = resp_id
                        chain_response.session_id = prompt.session_id
                        chain_response.choices.append(response_choice)
                        yield "data: " + str(chain_response.json()) + "\n\n"
                        resp_str = last_content
                        logger.debug(f"Using last AI message content as the final response: {last_content}")

                    snapshot = await app.agent.graph.aget_state(config)
                    # If there is a snapshot ask the user for return confirmation
                    if snapshot.next:
                        user_confirmation = "Do you approve of the process the return? Type 'y' to continue; otherwise, explain your requested changed."
                        chain_response = ChainResponse()
                        response_choice = ChainResponseChoices(
                            index=0,
                            message=Message(
                                role="assistant",
                                content=user_confirmation
                            )
                        )
                        chain_response.id = resp_id
                        chain_response.session_id = prompt.session_id
                        chain_response.choices.append(response_choice)
                        logger.debug(response_choice)
                        yield "data: " + str(chain_response.json()) + "\n\n"
                    # Check for the interrupt
                except asyncio.TimeoutError as te:
                    logger.info("This issue may occur if the LLM takes longer to respond. The timeout duration can be configured using the environment variable GRAPH_TIMEOUT_IN_SEC.")
                    logger.error(f"Graph Timeout Error. Error details: {te}")
                    is_exception = True
                except GraphRecursionError as ge:
                    logger.error(f"Graph Recursion Error. Error details: {ge}")
                    is_exception = True

            except AttributeError as attr_err:
                # Catch any specific attribute errors and log them
                logger.error(f"AttributeError: {attr_err}")
                print_exc()
                is_exception = True
            except asyncio.CancelledError as e:
                logger.error(f"Task was cancelled. Details: {e}")
                print_exc()
                is_exception = True
            except Exception as e:
                logger.error(f"Sending empty response. Unexpected error in response_generator: {e}")
                print_exc()
                is_exception = True

            if is_exception:
                logger.error("Sending back fallback responses since an exception was raised.")
                is_exception = False
                for data in fallback_response_generator(sentence=random.choice(FALLBACK_RESPONSES), session_id=prompt.session_id):
                    yield data

            chain_response = ChainResponse()
            # Initialize content with space to overwrite default response
            response_choice = ChainResponseChoices(
                        index=0,
                        message=Message(
                            role="assistant",
                            content=' '
                        ),
                        finish_reason="[DONE]"
                    )

            logger.info(f"Conversation saved:\nSession ID: {prompt.session_id}\nQuery: {last_user_message}\nResponse: {resp_str}")
            app.session_manager.save_conversation(
                prompt.session_id,
                prompt.user_id or "",
                [
                    {"role": "user", "content": user_query, "timestamp": f"{user_query_timestamp}"},
                    {"role": "assistant", "content": resp_str, "timestamp": f"{time.time()}"},
                ],
            )

            chain_response.id = resp_id
            chain_response.session_id = prompt.session_id
            chain_response.choices.append(response_choice)
            logger.debug(response_choice)
            yield "data: " + str(chain_response.json()) + "\n\n"

        return StreamingResponse(response_generator(), media_type="text/event-stream")

    # Catch any unhandled exceptions
    except asyncio.CancelledError as e:
        # Handle the cancellation gracefully
        logger.error("Unhandled Server interruption before response completion. Details: {e}")
        print_exc()
        return StreamingResponse(fallback_response_generator(sentence=random.choice(FALLBACK_RESPONSES), session_id=prompt.session_id), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Unhandled Error from /generate endpoint. Error details: {e}")
        print_exc()
        return StreamingResponse(fallback_response_generator(sentence=random.choice(FALLBACK_RESPONSES), session_id=prompt.session_id), media_type="text/event-stream")


@app.post("/feedback/response", tags=["Feedback"], response_model=FeedbackResponse, responses={
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
@weave.op()
async def store_last_response_feedback(
    request: Request,
    feedback: FeedbackRequest,
) -> FeedbackResponse:
    """Store user feedback for the last response in a conversation session."""
    try:
        logger.info(f"Storing user feedback for last response for session {feedback.session_id}")
        app.session_manager.response_feedback(feedback.session_id, feedback.feedback)
        return FeedbackResponse(message="Response feedback saved successfully")
    except Exception as e:
        logger.error(f"Error in GET /feedback/response endpoint. Error details: {e}")
        return FeedbackResponse(message="Failed to store response feedback")
