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
Quick Actions Services Package

This package provides intelligent quick actions and suggestions for
the warehouse operational assistant.
"""

from .smart_quick_actions import (
    SmartQuickActionsService,
    QuickAction,
    ActionContext,
    ActionType,
    ActionPriority,
    get_smart_quick_actions_service,
)

__all__ = [
    "SmartQuickActionsService",
    "QuickAction",
    "ActionContext",
    "ActionType",
    "ActionPriority",
    "get_smart_quick_actions_service",
]
