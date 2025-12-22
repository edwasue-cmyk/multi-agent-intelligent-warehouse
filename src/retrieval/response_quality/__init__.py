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
Response Quality Control Module for Warehouse Operational Assistant

Provides comprehensive response validation, quality assessment, and user experience
enhancements including confidence indicators, source attribution, and personalization.
"""

from .response_validator import (
    ResponseValidator,
    ResponseValidation,
    EnhancedResponse,
    SourceAttribution,
    ConfidenceIndicator,
    ConfidenceLevel,
    ResponseQuality,
    UserRole,
    get_response_validator
)

from .response_enhancer import (
    ResponseEnhancementService,
    AgentResponse,
    EnhancedAgentResponse,
    get_response_enhancer
)

from .ux_analytics import (
    UXAnalyticsService,
    UXMetric,
    UXTrend,
    UserExperienceReport,
    MetricType,
    get_ux_analytics
)

__all__ = [
    # Response Validator
    "ResponseValidator",
    "ResponseValidation",
    "EnhancedResponse",
    "SourceAttribution",
    "ConfidenceIndicator",
    "ConfidenceLevel",
    "ResponseQuality",
    "UserRole",
    "get_response_validator",
    
    # Response Enhancer
    "ResponseEnhancementService",
    "AgentResponse",
    "EnhancedAgentResponse",
    "get_response_enhancer",
    
    # UX Analytics
    "UXAnalyticsService",
    "UXMetric",
    "UXTrend",
    "UserExperienceReport",
    "MetricType",
    "get_ux_analytics"
]
