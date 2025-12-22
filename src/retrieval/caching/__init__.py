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
Caching Module for Warehouse Operational Assistant

Provides intelligent Redis caching for SQL results, evidence packs, vector searches,
and query preprocessing with configurable TTL, monitoring, and eviction policies.
"""

from .redis_cache_service import (
    RedisCacheService,
    CacheType,
    CacheConfig,
    CacheMetrics,
    CacheEntry,
    CachePolicy,
    get_cache_service
)

from .cache_manager import (
    CacheManager,
    CachePolicy as ManagerCachePolicy,
    CacheWarmingRule,
    EvictionStrategy,
    get_cache_manager
)

from .cache_integration import (
    CachedQueryProcessor,
    CacheIntegrationConfig,
    get_cached_query_processor
)

__all__ = [
    # Redis Cache Service
    "RedisCacheService",
    "CacheType",
    "CacheConfig", 
    "CacheMetrics",
    "CacheEntry",
    "CachePolicy",
    "get_cache_service",
    
    # Cache Manager
    "CacheManager",
    "ManagerCachePolicy",
    "CacheWarmingRule", 
    "EvictionStrategy",
    "get_cache_manager",
    
    # Cache Integration
    "CachedQueryProcessor",
    "CacheIntegrationConfig",
    "get_cached_query_processor"
]
