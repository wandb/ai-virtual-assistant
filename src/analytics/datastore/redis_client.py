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

"""
Manager conversation history return relavant conversation history
based on session_id using redis
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
from src.common.utils import get_config

# To store and fetch conversation we use database 0
DEFAULT_DB_CONVERSATION = "0"
# To store sentiment and summary we use database 1
DEFAULT_DB_SUMMARY = "1"
# To store feedback we use database 2
DEFAULT_DB_FEEDBACK = "2"


class RedisClient:
    def __init__(self) -> None:
        host, port = get_config().database.url.split(":")
        # db = config.get("db", None)  or DEFAULT_DB
        print(f"Host: {host}, Port: {port}")
        # Redis client to get and store conversation
        self.redis_client_conversation = redis.Redis(
            host=host, port=port, db=DEFAULT_DB_CONVERSATION, decode_responses=True
        )
        # Redis client to get and store summary
        self.redis_client_summary = redis.Redis(
            host=host, port=port, db=DEFAULT_DB_SUMMARY, decode_responses=True
        )
        self.redis_client_feedback = redis.Redis(
            host=host, port=port, db=DEFAULT_DB_FEEDBACK, decode_responses=True
        )


    def get_conversation(self, session_id: str) -> List:
        """Retrieve the entire conversation history from Redis as a list"""

        conversation_hist = self.redis_client_conversation.lrange(
            f"{session_id}:conversation_hist", 0, -1
        )
        return [json.loads(conv) for conv in conversation_hist]


    def get_k_conversation(self, session_id: str, k_turn: Optional[int] = None) -> List:
        """Retrieve the last k conversations from Redis"""

        # TODO: Evaluate this implementation
        if k_turn is None:
            k_turn = -1
        conversation_hist = self.redis_client_conversation.lrange(
            f"{session_id}:conversation_hist", -k_turn, -1
        )
        return [json.loads(conv) for conv in conversation_hist]


    def save_conversation(
        self, session_id: str, user_id: Optional[str], conversation: List
    ) -> bool:
        try:
            # Store each conversation entry as a JSON string in a Redis list
            for conv in conversation:
                self.redis_client_conversation.rpush(
                    f"{session_id}:conversation_hist", json.dumps(conv)
                )

            # Store user_id and last conversation time as separate keys
            if user_id:
                self.redis_client_conversation.set(f"{session_id}:user_id", user_id)
            self.redis_client_conversation.set(
                f"{session_id}:last_conversation_time", f"{time.time()}"
            )

            return True
        except Exception as e:
            print(f"Failed to ingest document due to exception {e}")
            return False


    def is_session(self, session_id: str) -> bool:
        """Check if session_id already exist in database"""
        return self.redis_client_conversation.exists(session_id)


    def list_sessions_for_user(self, user_id) -> Dict[str, Any]:
        """
        List all session IDs for a given user ID along with start and end times.

        Parameters:
            user_id (str): The user ID to filter sessions by.

        Returns:
            list: A list of dictionaries containing session ID, start time, and end time.
        """
        sessions = []
        # TODO: Optimize this, instead of traversing over all values
        # we can maintain a another list with user: [session] mapping
        for key in self.redis_client_conversation.scan_iter("*:user_id"):
            if self.redis_client_conversation.get(key) == user_id:
                session_id = key.split(":")[0]
                conversation_hist = self.redis_client_conversation.lrange(
                    f"{session_id}:conversation_hist", 0, -1
                )

                # Convert conversation history from JSON strings to dictionaries
                conversation_hist = [json.loads(conv) for conv in conversation_hist]

                sessions.append(
                    {
                        "session_id": session_id,
                        "start_time": (
                            datetime.fromtimestamp(
                                float(conversation_hist[0].get("timestamp"))
                            )
                            if conversation_hist
                            else None
                        ),
                        "end_time": (
                            datetime.fromtimestamp(
                                float(conversation_hist[-1].get("timestamp"))
                            )
                            if conversation_hist
                            else None
                        ),
                    }
                )
        return sessions


    def get_session_summary_and_sentiment(self, session_id):
        session_info = {}
        if self.redis_client_summary.exists(f"{session_id}:summary"):
            session_info["summary"] = self.redis_client_summary.get(f"{session_id}:summary")
            session_info["sentiment"] = self.redis_client_summary.get(f"{session_id}:sentiment")
            session_info["start_time"] = datetime.fromtimestamp(float(self.redis_client_summary.get(f"{session_id}:start_time")))
            session_info["end_time"] = datetime.fromtimestamp(float(self.redis_client_summary.get(f"{session_id}:end_time")))
        return session_info

    def save_summary_and_sentiment(self, session_id, session_info):
        """Save the summary, sentiment in separate dict"""

        self.redis_client_summary.set(f"{session_id}:summary", session_info.get("summary"))
        self.redis_client_summary.set(f"{session_id}:sentiment", session_info.get("sentiment"))
        self.redis_client_summary.set(f"{session_id}:start_time", session_info.get("start_time"))
        self.redis_client_summary.set(f"{session_id}:end_time", session_info.get("end_time"))


    def get_query_sentiment(self, session_id):
        session_info = {}
        if self.redis_client_summary.exists(f"{session_id}:conversation_hist"):
            session_info["messages"] = json.loads(self.redis_client_summary.get(f"{session_id}:conversation_hist"))
            session_info["start_time"] = datetime.fromtimestamp(float(self.redis_client_summary.get(f"{session_id}:start_time")))
            session_info["end_time"] = datetime.fromtimestamp(float(self.redis_client_summary.get(f"{session_id}:end_time")))
        return session_info


    def save_query_sentiment(self, session_id, session_info):
        self.redis_client_summary.set(f"{session_id}:conversation_hist", json.dumps(session_info.get("messages")))
        self.redis_client_summary.set(f"{session_id}:start_time", session_info.get("start_time"))
        self.redis_client_summary.set(f"{session_id}:end_time", session_info.get("end_time"))


    def save_sentiment_feedback(self, session_id: str, sentiment_feedback: float):
        """Save sentiment feedback"""
        return self.redis_client_feedback.set(f"{session_id}:sentiment", sentiment_feedback)


    def save_summary_feedback(self, session_id: str, summary_feedback: float):
        """Save summary feedback"""
        return self.redis_client_feedback.set(f"{session_id}:summary", summary_feedback)


    def save_session_feedback(self, session_id: str, session_feedback: float):
        """Save summary feedback"""
        return self.redis_client_feedback.set(f"{session_id}:session", session_feedback)
