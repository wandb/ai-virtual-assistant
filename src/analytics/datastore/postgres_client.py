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
based on session_id using postgres database
"""

from typing import Optional, List
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import psycopg2.extras
import psycopg2
import logging
import json
import os

from src.common.utils import get_config

# TODO: Move this config to __init__ method
db_user = os.environ.get("POSTGRES_USER")
db_password = os.environ.get("POSTGRES_PASSWORD")
db_name = os.environ.get("POSTGRES_DB")

logger = logging.getLogger(__name__)

settings = get_config()
# Postgres connection URL
DATABASE_URL = f"postgresql://{db_user}:{db_password}@{settings.database.url}/{db_name}?sslmode=disable"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

class ConversationHistory(Base):
    __tablename__ = 'conversation_history'

    session_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    last_conversation_time = Column(DateTime)
    start_conversation_time = Column(DateTime)
    conversation_data = Column(JSON)


class Summary(Base):
    __tablename__ = 'summary'

    session_id = Column(String, primary_key=True)
    start_time = Column(String)
    end_time = Column(String)
    sentiment = Column(String)
    summary = Column(String)
    conversation_data = Column(JSON)

class Feedback(Base):
    __tablename__ = 'feedback'

    session_id = Column(String, primary_key=True)
    sentiment = Column(Float) # Store feedback of sentiment
    summary = Column(Float) # Store feedback of summary
    session = Column(Float) # Store feedback of session

class PostgresClient:
    def __init__(self):
        self.engine = engine
        Base.metadata.create_all(self.engine)

    def store_conversation(self, session_id: str, user_id: Optional[str], conversation_history: list, last_conversation_time: str, start_conversation_time: str):
        session = Session()
        try:
            # Store last_conversation_time and start_conversation_time in datetime format for easy filtering
            conversation = ConversationHistory(
                session_id=session_id,
                user_id=user_id if user_id else None,
                last_conversation_time=datetime.fromtimestamp(float(last_conversation_time)),
                start_conversation_time=datetime.fromtimestamp(float(start_conversation_time)),
                conversation_data=json.dumps(conversation_history)
            )
            session.merge(conversation)
            session.commit()
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            session.rollback()
        finally:
            session.close()


    def get_conversation(self, session_id: str):
        session = Session()
        try:
            conversation = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            if conversation:
                return json.loads(conversation.conversation_data)
            raise ValueError(f"Session with ID {session_id} not found")

        except ValueError as ve:
            # Log the ValueError and propagate it
            logger.warning(f"{ve}")
            raise  # Propagate the ValueError to the caller
        except Exception as e:
            logger.error(f"Error fetching conversation: {e}")
            raise
        finally:
            session.close()


    def fetch_conversation(self, session_id: str):
        session = Session()
        try:
            conversation = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            if conversation:
                return {
                    'session_id': conversation.session_id,
                    'user_id': conversation.user_id,
                    'last_conversation_time': conversation.last_conversation_time,
                    'conversation_history': json.loads(conversation.conversation_data)
                }
            return None
        except Exception as e:
            logger.warning(f"Error fetching conversation: {e}")
            return None
        finally:
            session.close()


    def is_session(self, session_id: str) -> bool:
        session = Session()
        try:
            # Query to check if the session_id exists
            exists = session.query(ConversationHistory.session_id).filter_by(session_id=session_id).first() is not None
            return exists
        except Exception as e:
            logger.info(f"Error checking session: {e}")
            return False
        finally:
            session.close()


    def list_sessions_for_user(self, user_id: str):
        session = Session()
        try:
            # Query to get all session IDs for the user
            conversations = session.query(ConversationHistory).filter_by(user_id=user_id).all()

            result = []
            for conv in conversations:
                conversation_hist = json.loads(conv.conversation_data)
                if conversation_hist:
                    start_time = datetime.fromtimestamp(float(conversation_hist[0].get("timestamp"))) if conversation_hist[0].get("timestamp") else None
                    end_time = datetime.fromtimestamp(float(conversation_hist[-1].get("timestamp"))) if conversation_hist[-1].get("timestamp") else None
                else:
                    start_time = None
                    end_time = None

                result.append({
                    "session_id": conv.session_id,
                    "start_time": start_time,
                    "end_time": end_time
                })

            return result
        except Exception as e:
            logger.info(f"Error listing sessions for user: {e}")
            return []
        finally:
            session.close()


    def save_summary_and_sentiment(self, session_id, session_info):
        """Save the summary, sentiment in PostgreSQL"""
        session = Session()
        try:
            summary = Summary(
                session_id=session_id,
                start_time=session_info.get("start_time"),
                end_time=session_info.get("end_time"),
                sentiment=session_info.get("sentiment"),
                summary=session_info.get("summary"),
            )
            session.merge(summary)
            session.commit()
        except Exception as e:
            logger.info(f"Error saving summary and sentiment: {e}")
            session.rollback()
        finally:
            session.close()

    def get_session_summary_and_sentiment(self, session_id):
        """Retrieve summary and sentiment from PostgreSQL or Redis"""
        session = Session()
        try:
            # Check PostgreSQL first
            summary = session.query(Summary).filter_by(session_id=session_id).first()
            if summary:
                return {
                    "summary": summary.summary,
                    "sentiment": summary.sentiment,
                    "start_time": summary.start_time,
                    "end_time": summary.end_time
                }
            return {}
        except Exception as e:
            logger.info(f"Error retrieving session summary and sentiment: {e}")
            return {}
        finally:
            session.close()


    def save_query_sentiment(self, session_id, session_info):
        """Save query sentiment in PostgreSQL"""
        session = Session()
        try:
            query_sentiment = Summary(
                session_id=session_id,
                start_time=session_info.get("start_time"),
                end_time=session_info.get("end_time"),
                conversation_data=json.dumps(session_info.get("messages"))
            )
            session.merge(query_sentiment)
            session.commit()
        except Exception as e:
            logger.info(f"Error saving query sentiment: {e}")
            session.rollback()
        finally:
            session.close()

    def get_query_sentiment(self, session_id):
        """Retrieve query sentiment from PostgreSQL"""
        session = Session()
        try:
            query_sentiment = session.query(Summary).filter_by(session_id=session_id).first()
            if query_sentiment.conversation_data:
                return {
                    "messages": json.loads(query_sentiment.conversation_data),
                    "start_time": query_sentiment.start_time,
                    "end_time": query_sentiment.end_time
                }
            return {}
        except Exception as e:
            logger.info(f"Error retrieving query sentiment: {e}")
            return {}
        finally:
            session.close()


    def save_sentiment_feedback(self, session_id: str, sentiment_feedback: float):
        """Retrieve query sentiment from PostgreSQL"""
        session = Session()
        try:
            feedback = Feedback(
                session_id=session_id,
                sentiment=sentiment_feedback,
            )
            session.merge(feedback)
            session.commit()
        except Exception as e:
            logger.info(f"Error while saving sentiment feedback : {e}")
        finally:
            session.close()

    def save_summary_feedback(self, session_id: str, summary_feedback: float):
        session = Session()
        try:
            feedback = Feedback(
                session_id=session_id,
                summary=summary_feedback,
            )
            session.merge(feedback)
            session.commit()
        except Exception as e:
            logger.info(f"Error while saving sentiment feedback : {e}")
        finally:
            session.close()

    def save_session_feedback(self, session_id: str, session_feedback: float):
        session = Session()
        try:
            feedback = Feedback(
                session_id=session_id,
                session=session_feedback,
            )
            session.merge(feedback)
            session.commit()
        except Exception as e:
            logger.info(f"Error while saving session feedback : {e}")
            return {}
        finally:
            session.close()

    def get_purchase_history(self, user_id: str) -> List[str]:
        """Use this to retrieve the user's purchase history."""

        # TODO: Add filter logic based on time. Like product pruchased in last 5 days
        SQL_QUERY = f"""
        SELECT *
        FROM customer_data
        WHERE customer_id={user_id};
        """
        host, port = settings.database.url.split(":")

        db_params = {
            'dbname': os.environ.get("CUSTOMER_DATA_DB"),
            'user': db_user,
            'password': db_password,
            'host': host,
            'port': port
        }

        # Using context manager for connection and cursor
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(SQL_QUERY)
                result = cur.fetchall()

        # Returning result as a list of dictionaries
        return [dict(row) for row in result]

    def get_conversations_in_last_h_hours(self, hours: int):
        # Calculate the time threshold
        threshold_time = datetime.now() - timedelta(hours=hours)

        # Create a session
        session = Session()

        # Query conversations that occurred in the last h hours
        conversations = session.query(ConversationHistory).filter(
            ConversationHistory.last_conversation_time >= threshold_time
        ).all()

        # Close the session
        session.close()

        result = []

        for conv in conversations:
            result.append({
                "session_id": conv.session_id,
                "user_id": conv.user_id,
                "start_conversation_time": conv.start_conversation_time,
                "last_conversation_time": conv.last_conversation_time
            })
        return result
