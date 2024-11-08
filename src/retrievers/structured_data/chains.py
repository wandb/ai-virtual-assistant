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

""" Retriever pipeline for extracting data from structured information"""
import logging
from typing import Any, Dict, List

from pandasai import Agent as PandasAI_Agent
from pandasai.responses.response_parser import ResponseParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from src.retrievers.structured_data.connector import get_postgres_connector
from src.retrievers.base import BaseExample
from src.common.utils import get_config, get_prompts
from src.retrievers.structured_data.pandasai.llms.nv_aiplay import NVIDIA as PandasAI_NVIDIA

logger = logging.getLogger(__name__)
settings = get_config()


class PandasDataFrame(ResponseParser):
    """Returns Pandas Dataframe instead of SmartDataFrame"""

    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        return result["value"]


class CSVChatbot(BaseExample):
    """RAG example showcasing CSV parsing using Pandas AI Agent"""

    def ingest_docs(self, filepath: str, filename: str):
        """Ingest documents to the VectorDB."""

        raise NotImplementedError("Canonical RAG only supports document retrieval")

    def document_search(self, content: str, num_docs: int, user_id: str = None, conv_history: Dict[str, str] = {}) -> List[Dict[str, Any]]:
        """Execute a Document Search."""

        logger.info("Using document_search to fetch response from database as text")
        postgres_connector = None  # Initialize connector

        try:
            logger.info("Using document_search to fetch response from database as text")
            if user_id:
                postgres_connector = get_postgres_connector(user_id)
            else:
                logger.warning("Enter a proper User ID")
                return [{"content": "No response generated, make to give a proper User ID."}]

            # TODO: Pass conv history to the LLM
            llm_data_retrieval = PandasAI_NVIDIA(temperature=0.2, model=settings.llm.model_name_pandas_ai)

            config_data_retrieval = {"llm": llm_data_retrieval, "response_parser": PandasDataFrame, "max_retries": 1, "enable_cache": False}
            agent_data_retrieval = PandasAI_Agent([postgres_connector], config=config_data_retrieval, memory_size=20)

            prompt_config = get_prompts().get("prompts")
            
            data_retrieval_prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(prompt_config.get("data_retrieval_template", [])),
                    HumanMessagePromptTemplate.from_template("{query}"),
                ],
                input_variables=["description", "instructions", "query"],
            )


            chat_prompt = data_retrieval_prompt.format_prompt(
                    description=prompt_config.get("dataframe_prompts").get("customer_data").get("description"),
                    instructions=prompt_config.get("dataframe_prompts").get("customer_data").get("instructions"),
                    query=content,
                ).to_string()
            
            result_df = agent_data_retrieval.chat(
                chat_prompt
            )
            logger.info("Result Data Frame: %s", result_df)
            if not result_df:
                logger.warning("Retrieval failed to get any relevant context")
                return [{"content": "No response generated from LLM, make sure your query is relavent to the ingested document."}]

            result_df = str(result_df)
            return [{"content": result_df}]
        except Exception as e:
            logger.error("An error occurred during document search: %s", str(e))
            raise  # Re-raise the exception after logging
        
        finally:
            if postgres_connector:
                postgres_connector._connection._dbapi_connection.close()
                postgres_connector._connection.close()
                postgres_connector._engine.dispose()
                import gc
                gc.collect()
                logger.info("Postgres connector deleted.")

    def get_documents(self) -> List[str]:
        """Retrieves filenames stored in the vector store."""
        logger.error("get_documents not implemented")
        return True

    def delete_documents(self, filenames: List[str]):
        """Delete documents from the vector index."""
        logger.error("delete_documents not implemented")
        return True
