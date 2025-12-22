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
IoT Sensor Integration for Warehouse Operational Assistant.

This module provides adapters for IoT sensor integration including:
- Equipment monitoring sensors
- Environmental sensors
- Safety sensors
- Asset tracking sensors

Each adapter implements a common interface for seamless integration.
"""

from .base import BaseIoTAdapter, IoTConnectionError, IoTDataError
from .equipment_monitor import EquipmentMonitorAdapter
from .environmental import EnvironmentalSensorAdapter
from .safety_sensors import SafetySensorAdapter
from .asset_tracking import AssetTrackingAdapter
from .factory import IoTAdapterFactory

__all__ = [
    'BaseIoTAdapter',
    'IoTConnectionError', 
    'IoTDataError',
    'EquipmentMonitorAdapter',
    'EnvironmentalSensorAdapter',
    'SafetySensorAdapter',
    'AssetTrackingAdapter',
    'IoTAdapterFactory'
]
