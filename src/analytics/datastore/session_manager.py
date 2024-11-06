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
based on session_id
"""

from typing import Any, Dict, List, Optional

from src.analytics.datastore.postgres_client import PostgresClient
from src.analytics.datastore.redis_client import RedisClient
from src.common.utils import get_config


class SessionManager:
    """
    Store the conversation between user and assistant, it's stored in format
    {"session_id": {"user_id": "", "conversation_hist": [{"role": "user/assistant", "content": "", "timestamp": ""}], "last_conversation_time: ""}}

    Store summary of conversation
    {"session_id": {"summary": "", "sentiment": "", "start_time: "", "end_time: ""}}
    """

    def __init__(self, *args, **kwargs) -> None:
        db_name = get_config().database.name
        if db_name == "redis":
            print("Using Redis client for user history")
            self.memory = RedisClient()
        elif db_name == "postgres":
            print("Using postgres for user history")
            self.memory = PostgresClient()
        else:
            raise ValueError(
                f"{db_name} in not supported. Supported type redis, postgres"
            )

    # TODO: Create API to get last k conversation from database instead of returning everything
    def get_conversation(self, session_id: str) -> List:
        return self.memory.get_conversation(session_id)

    def get_k_conversation(self, session_id: str, k_turn: Optional[int] = None) -> List:
        return self.memory.get_k_conversation(session_id, k_turn)

    def save_conversation(
        self, session_id: str, user_id: Optional[str], conversation: List
    ) -> bool:
        return self.memory.save_conversation(session_id, user_id, conversation)

    def is_session(self, session_id: str) -> bool:
        """Check if session_id already exist in database"""
        return self.memory.is_session(session_id)

    def list_sessions_for_user(self, user_id) -> Dict[str, Any]:
        """
        List all session IDs for a given user ID along with start and end times.
        """
        return self.memory.list_sessions_for_user(user_id)

    def get_session_summary_and_sentiment(self, session_id):
        return self.memory.get_session_summary_and_sentiment(session_id)

    def save_summary_and_sentiment(self, session_id, session_info):
        """Save the summary, sentiment in separate dict"""

        return self.memory.save_summary_and_sentiment(session_id, session_info)

    def get_query_sentiment(self, session_id):

        return self.memory.get_query_sentiment(session_id)

    def save_query_sentiment(self, session_id, conversation_hist):

        return self.memory.save_query_sentiment(session_id, conversation_hist)

    def save_sentiment_feedback(self, session_id: str, sentiment_feedback: float):
        """Save sentiment feedback"""
        return self.memory.save_sentiment_feedback(session_id, sentiment_feedback)


    def save_summary_feedback(self, session_id: str, summary_feedback: float):
        return self.memory.save_summary_feedback(session_id, summary_feedback)


    def save_session_feedback(self, session_id: str, session_feedback: float):
        return self.memory.save_session_feedback(session_id, session_feedback)


    def get_purchase_history(self, user_id: str) -> List[str]:
        """Use this to retrieve the user's purchase history."""
        return self.memory.get_purchase_history(user_id)


    def get_conversations_in_last_h_hours(self, hours: int):
        return self.memory.get_conversations_in_last_h_hours(hours)
