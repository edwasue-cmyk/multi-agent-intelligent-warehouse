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
Evidence Services Package

This package provides comprehensive evidence collection and context synthesis
capabilities for the warehouse operational assistant.
"""

from .evidence_collector import (
    EvidenceCollector,
    Evidence,
    EvidenceContext,
    EvidenceType,
    EvidenceSource,
    EvidenceQuality,
    get_evidence_collector,
)

from .evidence_integration import (
    EvidenceIntegrationService,
    EnhancedResponse,
    get_evidence_integration_service,
)

__all__ = [
    "EvidenceCollector",
    "Evidence",
    "EvidenceContext",
    "EvidenceType",
    "EvidenceSource",
    "EvidenceQuality",
    "get_evidence_collector",
    "EvidenceIntegrationService",
    "EnhancedResponse",
    "get_evidence_integration_service",
]
