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
import numpy as np
from typing import List


class NVIDIAEmbeddingsWrapper:
    def __init__(self, nvidia_embeddings):
        self.nvidia_embeddings = nvidia_embeddings

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        # Convert each embedding from embed_query to np.array
        # return [np.array(embedding) for embedding in self.nvidia_embeddings.embed_query(queries)]
        return list(map(np.array, [self.nvidia_embeddings.embed_query(query) for query in queries]))

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        # Convert each embedding from embed_documents to np.array
        return list(map(np.array, self.nvidia_embeddings.embed_documents(documents)))