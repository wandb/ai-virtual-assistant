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
from urllib.parse import urlparse
from src.common.utils import get_config
from pandasai.connectors import PostgreSQLConnector

def get_postgres_connector(customer_id: str) -> PostgreSQLConnector:

    app_database_url = get_config().database.url

    # Parse the URL
    parsed_url = urlparse(f"//{app_database_url}", scheme='postgres')

    # Extract host and port
    host = parsed_url.hostname
    port = parsed_url.port

    config = {
        "host": host,
        "port": port,
        "database": os.getenv('POSTGRES_DB', None),
        "username": os.getenv('POSTGRES_USER', None),
        "password": os.getenv('POSTGRES_PASSWORD', None),
        "table": "customer_data",
        "where": [
            ["customer_id", "=", customer_id],
        ],
    }
    return PostgreSQLConnector(config=config)