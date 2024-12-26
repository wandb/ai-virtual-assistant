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

import logging
import os
import weave
from typing import Any, Dict, List
from traceback import print_exc

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from pydantic import BaseModel, Field

from src.retrievers.base import BaseExample
from src.common.utils import (
    create_vectorstore_langchain,
    del_docs_vectorstore_langchain,
    get_config,
    get_docs_vectorstore_langchain,
    get_embedding_model,
    get_prompts,
    get_llm,
    get_text_splitter,
    get_vectorstore,
    get_ranking_model
)

logger = logging.getLogger(__name__)
document_embedder = get_embedding_model()
text_splitter = None
settings = get_config()
prompts = get_prompts()
vector_db_top_k = int(os.environ.get(f"VECTOR_DB_TOPK", 40))

try:
    vectorstore = create_vectorstore_langchain(document_embedder=document_embedder)
except Exception as e:
    vectorstore = None
    logger.info(f"Unable to connect to vector store during initialization: {e}")


class UnstructuredRetriever(BaseExample):
    @weave.op()
    def ingest_docs(self, filepath: str, filename: str) -> None:
        """Ingests documents to the VectorDB.
        It's called when the POST endpoint of `/documents` API is invoked.

        Args:
            filepath (str): The path to the document file.
            filename (str): The name of the document file.

        Raises:
            ValueError: If there's an error during document ingestion or the file format is not supported.
        """
        if not filename.endswith((".txt", ".pdf", ".md")):
            raise ValueError(f"{filename} is not a valid Text, PDF or Markdown file")
        try:
            # Load raw documents from the directory
            _path = filepath
            raw_documents = UnstructuredFileLoader(_path).load()

            if raw_documents:
                global text_splitter
                # Get text splitter instance, it is selected based on environment variable APP_TEXTSPLITTER_MODELNAME
                # tokenizer dimension of text splitter should be same as embedding model
                if not text_splitter:
                    text_splitter = get_text_splitter()

                # split documents based on configuration provided
                documents = text_splitter.split_documents(raw_documents)
                vs = get_vectorstore(vectorstore, document_embedder)
                # ingest documents into vectorstore
                vs.add_documents(documents)
            else:
                logger.warning("No documents available to process!")
        except Exception as e:
            logger.error(f"Failed to ingest document due to exception {e}")
            raise ValueError("Failed to upload document. Please upload an unstructured text document.")

    @weave.op()
    def document_search(self, content: str, num_docs: int, conv_history: Dict[str, str] = {}) -> List[Dict[str, Any]]:
        """Search for the most relevant documents for the given search parameters.
        It's called when the `/search` API is invoked.

        Args:
            content (str): Query to be searched from vectorstore.
            num_docs (int): Number of similar docs to be retrieved from vectorstore.
        """

        logger.info(f"Searching relevant document for the query: {content}")

        try:
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs == None:
                logger.error(f"Vector store not initialized properly. Please check if the vector db is up and running")
                raise ValueError()

            docs = []
            ranker = get_ranking_model()
            top_k = vector_db_top_k if ranker else num_docs
            logger.info(f"Setting top k as: {top_k}.")
            retriever = vs.as_retriever(search_kwargs={"k": top_k}) # milvus does not support similarily threshold

            # Invoke query rewriting to decontextualize the query before sending to retriever pipeline if conv history is passed
            if conv_history:
                class Question(BaseModel):
                    question: str = Field(..., description="A standalone question which can be understood without the chat history")

                parsed_conv_history = [(msg.get("role"), msg.get("content")) for msg in conv_history]
                default_llm_kwargs = {"temperature": 0.2, "top_p": 0.7, "max_tokens": 1024}
                llm = get_llm(**default_llm_kwargs)
                llm = llm.with_structured_output(Question)
                query_rewriter_prompt = prompts.get("query_rewriting")
                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [("system", query_rewriter_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),]
                )
                q_prompt = contextualize_q_prompt | llm 
                logger.info(f"Query rewriter prompt: {contextualize_q_prompt}")
                response = q_prompt.invoke({"input": content, "chat_history": parsed_conv_history})
                content = response.question
                logger.info(f"Rewritten Query: {content}")
                if content.replace('"', "'") == "''" or len(content) == 0:
                    return []

            if ranker:
                logger.info(f"Narrowing the collection from {top_k} results and further narrowing it to {num_docs} with the reranker for rag chain.")
                # Update number of document to be retriever by ranker
                ranker.top_n = num_docs

                context_reranker = RunnableAssign({"context": lambda input: ranker.compress_documents(query=input['question'], documents=input['context'])})

                retriever = {"context": retriever, "question": RunnablePassthrough()} | context_reranker
                docs = retriever.invoke(content)
                resp = []
                for doc in docs.get("context"):
                    resp.append(
                            {
                                "source": os.path.basename(doc.metadata.get("source", "")),
                                "content": doc.page_content,
                                "score": doc.metadata.get("relevance_score", 0)
                            }
                    )
                return resp
            else:
                docs = retriever.invoke(content)
                resp = []
                for doc in docs:
                    resp.append(
                            {
                                "source": os.path.basename(doc.metadata.get("source", "")),
                                "content": doc.page_content,
                                "score": doc.metadata.get("relevance_score", 0)
                            }
                    )
                return resp

        except Exception as e:
            logger.warning(f"Failed to generate response due to exception {e}")
            print_exc()

        return []

    @weave.op()
    def get_documents(self) -> List[str]:
        """Retrieves filenames stored in the vector store.
        It's called when the GET endpoint of `/documents` API is invoked.

        Returns:
            List[str]: List of filenames ingested in vectorstore.
        """
        try:
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs:
                return get_docs_vectorstore_langchain(vs)
        except Exception as e:
            logger.error(f"Vectorstore not initialized. Error details: {e}")
        return []

    @weave.op()
    def delete_documents(self, filenames: List[str]) -> bool:
        """Delete documents from the vector index.
        It's called when the DELETE endpoint of `/documents` API is invoked.

        Args:
            filenames (List[str]): List of filenames to be deleted from vectorstore.
        """
        try:
            # Get vectorstore instance
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs:
                return del_docs_vectorstore_langchain(vs, filenames)
        except Exception as e:
            logger.error(f"Vectorstore not initialized. Error details: {e}")
        return False
