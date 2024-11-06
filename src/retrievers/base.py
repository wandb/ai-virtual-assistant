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

"""Base interface that all Retriever examples should implement."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseExample(ABC):

    @abstractmethod
    def document_search(self, content: str, num_docs: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_documents(self) -> List[str]:
        pass

    @abstractmethod
    def delete_documents(self, filenames: List[str]) -> bool:
        pass

    @abstractmethod
    def ingest_docs(self, data_dir: str, filename: str) -> None:
        pass