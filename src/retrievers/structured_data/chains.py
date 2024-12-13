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
import os
import weave
from typing import Any, Dict, List
from urllib.parse import urlparse
import pandas as pd
from src.retrievers.structured_data.vaanaai.vaana_base import VannaWrapper
from src.retrievers.base import BaseExample
from src.common.utils import get_config

logger = logging.getLogger(__name__)
settings = get_config()

# Load the vaana_client
vaana_client = VannaWrapper()
# Connect to the Postgress DB
app_database_url = get_config().database.url

# Parse the URL
parsed_url = urlparse(f"//{app_database_url}", scheme='postgres')

# Extract host and port
host = parsed_url.hostname
port = parsed_url.port

vaana_client.connect_to_postgres(
    host=parsed_url.hostname, 
    dbname=os.getenv("POSTGRES_DB",'customer_data'), 
    user=os.getenv('POSTGRES_USER', 'postgres_readonly'), 
    password= os.getenv('POSTGRES_PASSWORD', 'readonly_password'), 
    port=parsed_url.port
    )
# Do Training from static schmea
vaana_client.do_training(method="schema")

class CSVChatbot(BaseExample):
    """RAG example showcasing CSV parsing using Vaana AI Agent"""
    @weave.op()
    def ingest_docs(self, filepath: str, filename: str):
        """Ingest documents to the VectorDB."""

        raise NotImplementedError("Canonical RAG only supports document retrieval")

    @weave.op()
    def document_search(self, content: str, num_docs: int, user_id: str = None, conv_history: Dict[str, str] = {}) -> List[Dict[str, Any]]:
        """Execute a Document Search."""

        logger.info("Using document_search to fetch response from database as text")

        # Do training if the static_db_schema is empty
        vaana_client.do_training(method="ddl")

        try:
            logger.info("Using document_search to fetch response from database as text")
            if user_id:
                pass
            else:
                logger.warning("Enter a proper User ID")
                return [{"content": "No response generated, make to give a proper User ID."}]

            result_df = vaana_client.ask_query(question=content, user_id=user_id)
            
            logger.info("Result Data Frame: %s", result_df)
            if (
                (isinstance(result_df, pd.DataFrame) and (
                    (result_df.shape == (1, 1) and not bool(result_df.iloc[0, 0])) or result_df.empty
                )) or
                (isinstance(result_df, str) and result_df == "not valid sql") or
                (result_df is None)
                ):
                logger.warning("Retrieval failed to get any relevant context")
                raise Exception("No response generated from LLM.")

            result_df = str(result_df).strip()
            return [{"content": result_df}]
        except Exception as e:
            logger.error("An error occurred during document search: %s", str(e))
            raise  # Re-raise the exception after logging

    @weave.op()
    def get_documents(self) -> List[str]:
        """Retrieves filenames stored in the vector store."""
        logger.error("get_documents not implemented")
        return True

    @weave.op()
    def delete_documents(self, filenames: List[str]):
        """Delete documents from the vector index."""
        logger.error("delete_documents not implemented")
        return True
