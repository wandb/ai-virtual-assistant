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
import os
import logging
from datetime import datetime
from enum import Enum
from typing import Annotated, Generator, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from src.analytics.datastore.session_manager import SessionManager
from src.common.utils import get_config, get_llm, get_prompts

logger = logging.getLogger(__name__)
prompts = get_prompts()

# TODO get the default_kwargs from the Agent Server API
default_llm_kwargs = {"temperature": 0, "top_p": 0.7, "max_tokens": 1024}

# Initialize persist_data to determine whether data should be stored in the database.
persist_data = os.environ.get("PERSIST_DATA", "true").lower() == "true"

# Initialize session manager during startup
session_manager = None
try:
    session_manager = SessionManager()
except Exception as e:
    logger.info(f"Failed to connect to DB during init, due to exception {e}")


def get_database():
    """
    Connect to the database.
    """
    global session_manager
    try:
        if not session_manager:
            session_manager = SessionManager()

        return session_manager
    except Exception as e:
        logger.info(f"Error connecting to database: {e}")
        return None


def generate_summary(conversation_history):
    """
    Generate a summary of the conversation.

    Parameters:
        conversation_history (List): The conversation text.

    Returns:
        str: A summary of the conversation.
    """
    logger.info(f"conversation history: {conversation_history}")
    llm = get_llm(**default_llm_kwargs)
    prompt = prompts.get("summary_prompt", "")
    for turn in conversation_history:
        prompt += f"{turn['role']}: {turn['content']}\n"

    prompt += "\n\nSummary: "
    response = llm.invoke(prompt)

    return response.content


def generate_session_summary(session_id):
    # TODO: Check for corner cases like when session_id does not exist
    session_manager = get_database()

    # Check if summary already exists in database
    session_info = session_manager.get_session_summary_and_sentiment(session_id)
    if session_info and session_info.get("summary", None):
        return session_info

    # Generate summary and session info
    conversation_history = session_manager.get_conversation(session_id)
    summary = generate_summary(conversation_history)
    sentiment = generate_sentiment(conversation_history)

    if persist_data:
        # Save the summary and sentiment in database
        session_manager.save_summary_and_sentiment(
            session_id,
            {
                "summary": summary,
                "sentiment": sentiment,
                "start_time": conversation_history[0].get("timestamp", 0),
                "end_time": conversation_history[-1].get("timestamp", 0),
            }
        )
    return {
        "summary": summary,
        "sentiment": sentiment,
        "start_time": datetime.fromtimestamp(
            float(conversation_history[0].get("timestamp", 0))
        ),
        "end_time": datetime.fromtimestamp(
            float(conversation_history[-1].get("timestamp", 0))
        ),
    }


def fetch_user_conversation(user_id, start_time=None, end_time=None):
    """
    Fetch a user's conversation from the database.
    """
    try:
        # TODO: Use start time and end time to filter the data
        session_manager = get_database()
        conversations = session_manager.list_sessions_for_user(user_id)
        logger.info(f"Conversation: {conversations}")
        return conversations
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        return None


def generate_sentiment(conversation_history):
    # Define an Enum for the sentiment values
    class SentimentEnum(str, Enum):
        POSITIVE = "positive"
        NEUTRAL = "neutral"
        NEGATIVE = "negative"

    # Define the Pydantic model using the Enum
    class Sentiment(BaseModel):
        """Sentiment for conversation."""

        sentiment: SentimentEnum = Field(
            description="Relevant value 'positive', 'neutral' or 'negative'"
        )

    logger.info("Finding sentiment for conversation")
    llm = get_llm(**default_llm_kwargs)
    prompt = prompts.get("sentiment_prompt", "")
    for turn in conversation_history:
        prompt += f"{turn['role']}: {turn['content']}\n"

    llm_with_tool = llm.with_structured_output(Sentiment)

    response = llm_with_tool.invoke(prompt)
    sentiment = response.sentiment.value
    logger.info(f"Conversation classified as {sentiment}")
    return sentiment


def generate_sentiment_for_query(session_id):
    """Generate sentiment for user query and assistant response
    """

    logger.info("Fetching sentiment for queries")
    # Check if the sentiment is already identified in database, if yes return that
    session_manager = get_database()

    session_info = session_manager.get_query_sentiment(session_id)

    if session_info and session_info.get("messages", None):
        return {
        "messages": session_info.get("messages"),
            "session_info": {
                "session_id": session_id,
                "start_time": session_info.get("start_time"),
                "end_time": session_info.get("start_time"),
            },
        }

    class SentimentEnum(str, Enum):
        POSITIVE = "positive"
        NEUTRAL = "neutral"
        NEGATIVE = "negative"

    # Define the Pydantic model using the Enum
    class Sentiment(BaseModel):
        """Sentiment for conversation."""

        sentiment: SentimentEnum = Field(
            description="Relevant value 'positive', 'neutral' or 'negative'"
        )


    # Generate summary and session info
    conversation_history = session_manager.get_conversation(session_id)
    logger.info(f"Conversation history: {conversation_history}")

    logger.info("Finding sentiment for conversation")
    llm = get_llm(**default_llm_kwargs)

    llm_with_tool = llm.with_structured_output(Sentiment)

    messages = []
    # TODO: parallize this operation for faster response
    # Find sentiment for individual query and assistant response
    for turn in conversation_history:
        prompt = prompts.get("query_sentiment_prompt", "")
        prompt += f"{turn['role']}: {turn['content']}\n"

        response = llm_with_tool.invoke(prompt)
        sentiment = response.sentiment.value
        messages.append({
            "role": turn["role"],
            "content": turn["content"],
            "sentiment": sentiment,
        })

    session_info = {
        "messages": messages,
        "start_time": conversation_history[0].get("timestamp", 0),
        "end_time": conversation_history[-1].get("timestamp", 0),
    }
    if persist_data:
        # Save information before sending it to user
        session_manager.save_query_sentiment(session_id, session_info)
    return {
        "messages": messages,
            "session_info": {
                "session_id": session_id,
                "start_time": datetime.fromtimestamp(
                    float(conversation_history[0].get("timestamp", 0))
                ),
                "end_time": datetime.fromtimestamp(
                    float(conversation_history[-1].get("timestamp", 0))
                ),
            },
    }

