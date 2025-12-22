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
ERP Integration Adapters

This module provides adapters for integrating with various ERP systems
including SAP ECC, Oracle ERP, and other enterprise resource planning systems.
"""

from .base import BaseERPAdapter, ERPConnection, ERPResponse
from .sap_ecc import SAPECCAdapter
from .oracle_erp import OracleERPAdapter
from .factory import ERPAdapterFactory

__all__ = [
    "BaseERPAdapter",
    "ERPConnection", 
    "ERPResponse",
    "SAPECCAdapter",
    "OracleERPAdapter",
    "ERPAdapterFactory"
]
