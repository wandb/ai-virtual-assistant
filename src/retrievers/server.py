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

"""The definition of the Retrievers FASTAPI server."""

import os
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib
from inspect import getmembers, isclass
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel, Field, constr

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "Health",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {"name": "Core", "description": "Core APIs for ingestion and searching."},
    {"name": "Management", "description": "APIs for deleting and listing ingested files."},
]

# create the FastAPI server
app = FastAPI(title="Retriever API's for AI Virtual Assistant for Customer Service",
    description="This API schema describes all the retriever endpoints exposed for the AI Virtual Assistant for Customer Service NIM Blueprint",
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

class DocumentSearch(BaseModel):
    """Definition of the DocumentSearch API data type."""

    query: str = Field(description="The content or keywords to search for within documents.", max_length=131072, pattern=r'[\s\S]*', default="")
    top_k: int = Field(description="The maximum number of documents to return in the response.", default=4, ge=0, le=25, format="int64")
    user_id: Optional[str] = Field(description="An optional unique identifier for the customer.", default=None)
    conv_history: Optional[List[Dict[str, str]]] = Field(description="An optional conversation history for the customer.", default=[])

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

@app.on_event("startup")
def import_example() -> None:
    """
    Import the example class from the specified example file.
    The example directory is expected to have a python file where the example class is defined.
    """

    file_location = os.path.join(EXAMPLE_DIR, os.environ.get("EXAMPLE_PATH", "./"))

    for root, dirs, files in os.walk(file_location):
        for file in files:
            if not file.endswith(".py"):
                continue

            # Import the specified file dynamically
            spec = importlib.util.spec_from_file_location(name="example", location=os.path.join(root, file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Scan each class in the file to find one with the 3 implemented methods: ingest_docs, rag_chain and llm_chain
            for name, _ in getmembers(module, isclass):
                try:
                    cls = getattr(module, name)
                    if set(["ingest_docs"]).issubset(set(dir(cls))):
                        if name == "BaseExample":
                            continue
                        example = cls()
                        app.example = cls
                        return
                except:
                    raise ValueError(f"Class {name} is not implemented and could not be instantiated.")

    raise NotImplementedError(f"Could not find a valid example class in {file_location}")

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
def health_check():
    """
    Perform a Health Check

    Returns 200 when service is up. This does not check the health of downstream services.
    """

    response_message = "Service is up."
    return HealthResponse(message=response_message)


@app.post("/documents", tags=["Core"], responses={
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def upload_document(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Upload a document to the vector store."""

    if not file.filename:
        return JSONResponse(content={"message": "No files provided"}, status_code=200)

    try:
        upload_folder = "/tmp-data/uploaded_files"
        upload_file = os.path.basename(file.filename)
        if not upload_file:
            raise RuntimeError("Error parsing uploaded filename.")
        file_path = os.path.join(upload_folder, upload_file)
        uploads_dir = Path(upload_folder)
        uploads_dir.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        app.example().ingest_docs(file_path, upload_file)

        return JSONResponse(
            content={"message": "File uploaded successfully"}, status_code=200
        )

    except Exception as e:
        logger.error("Error from POST /documents endpoint. Ingestion of file: " + file.filename + " failed with error: " + str(e))
        return JSONResponse(
            content={"message": str(e)}, status_code=500
        )


@app.post("/search", tags=["Core"], response_model=DocumentSearchResponse, responses={
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def document_search(request: Request, data: DocumentSearch) -> Dict[str, List[Dict[str, Any]]]:
    """Search for the most relevant documents for the given search parameters."""

    try:
        example = app.example()
        if hasattr(example, "document_search") and callable(example.document_search):
            # This is needed as structured_rag needs user_id aka user
            if data.user_id:
                search_result = example.document_search(data.query, data.top_k, data.user_id, data.conv_history)
            else:
                search_result = example.document_search(data.query, data.top_k, data.conv_history)
            chunks = []
            for entry in search_result:
                content = entry.get("content", "")  # Default to empty string if "content" key doesn't exist
                source = entry.get("source", "")    # Default to empty string if "source" key doesn't exist
                score = entry.get("score", 0.0)     # Default to 0.0 if "score" key doesn't exist
                chunk = DocumentChunk(content=content, filename=source, document_id="", score=score)
                chunks.append(chunk)
            return DocumentSearchResponse(chunks=chunks)
        raise NotImplementedError("Example class has not implemented the document_search method.")

    except Exception as e:
        logger.error(f"Error from POST /search endpoint. Error details: {e}")
        return JSONResponse(content={"message": "Error occurred while searching documents."}, status_code=500)


@app.get("/documents", tags=["Management"], response_model=DocumentsResponse, responses={
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def get_documents(request: Request) -> DocumentsResponse:
    """List available documents."""
    try:
        example = app.example()
        if hasattr(example, "get_documents") and callable(example.get_documents):
            documents = example.get_documents()
            return DocumentsResponse(documents=documents)
        else:
            raise NotImplementedError("Example class has not implemented the get_documents method.")

    except Exception as e:
        logger.error(f"Error from GET /documents endpoint. Error details: {e}")
        return JSONResponse(content={"message": "Error occurred while fetching documents."}, status_code=500)


@app.delete("/documents", tags=["Management"], responses={
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def delete_document(request: Request, filename: str) -> JSONResponse:
    """Delete a document."""
    try:
        example = app.example()
        if hasattr(example, "delete_documents") and callable(example.delete_documents):
            status = example.delete_documents([filename])
            if not status:
                raise Exception(f"Error in deleting document {filename}")
            return JSONResponse(content={"message": f"Document {filename} deleted successfully"}, status_code=200)

        raise NotImplementedError("Example class has not implemented the delete_document method.")

    except Exception as e:
        logger.error(f"Error from DELETE /documents endpoint. Error details: {e}")
        return JSONResponse(content={"message": f"Error deleting document {filename}"}, status_code=500)
