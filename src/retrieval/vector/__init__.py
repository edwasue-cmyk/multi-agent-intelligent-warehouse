# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Vector Retrieval Module for Warehouse Operations

This module provides vector-based retrieval capabilities using Milvus
for semantic search over SOPs, manuals, and other unstructured content.
"""

from .milvus_retriever import MilvusRetriever
from .embedding_service import EmbeddingService
from .hybrid_ranker import HybridRanker
from .chunking_service import ChunkingService, Chunk, ChunkMetadata
from .enhanced_retriever import EnhancedVectorRetriever, EnhancedSearchResult, RetrievalConfig
from .evidence_scoring import EvidenceScoringEngine, EvidenceSource, EvidenceItem, EvidenceScore
from .clarifying_questions import ClarifyingQuestionsEngine, QuestionSet, ClarifyingQuestion, AmbiguityType, QuestionPriority

__all__ = [
    "MilvusRetriever",
    "EmbeddingService", 
    "HybridRanker",
    "ChunkingService",
    "Chunk",
    "ChunkMetadata",
    "EnhancedVectorRetriever",
    "EnhancedSearchResult",
    "RetrievalConfig",
    "EvidenceScoringEngine",
    "EvidenceSource",
    "EvidenceItem",
    "EvidenceScore",
    "ClarifyingQuestionsEngine",
    "QuestionSet",
    "ClarifyingQuestion",
    "AmbiguityType",
    "QuestionPriority"
]
