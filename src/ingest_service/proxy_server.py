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

""" This module is just a proxy server which calls agent container's health API """

import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# create the FastAPI server
app = FastAPI()


class HealthResponse(BaseModel):
    """Health check response"""

    message: str = Field(max_length=4096, pattern=r"[\s\S]*", default="")


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error occurred"}
                }
            },
        }
    },
)
def health_check():
    """
    Perform a Health Check

    Returns 200 when service is up. This does not check the health of downstream services.
    """

    response_message = "Service is up."
    return HealthResponse(message=response_message)
