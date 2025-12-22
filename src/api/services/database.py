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
Database Service for Warehouse Operational Assistant

This service provides database connection management and health checks.
"""

import os
import asyncpg
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)


async def get_database_connection():
    """
    Get a database connection as an async context manager.

    Returns:
        asyncpg.Connection: Database connection
    """
    # Get database URL from environment
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5435/warehouse_ops"
    )

    return asyncpg.connect(database_url)
