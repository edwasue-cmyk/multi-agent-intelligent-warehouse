#!/usr/bin/env python3
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
NVIDIA API Setup Script for Warehouse Operational Assistant

This script helps you configure your NVIDIA API keys for the LLM and embedding services.
"""

import os
import sys
from pathlib import Path

def setup_nvidia_api():
    """Interactive setup for NVIDIA API keys."""
    print("🚀 NVIDIA API Setup for Warehouse Operational Assistant")
    print("=" * 60)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Please run this script from the project root.")
        return False
    
    print("\n📋 Required NVIDIA Services:")
    print("1. LLM NIM (Llama 3 70B) - for reasoning and response generation")
    print("2. Embedding NIM (NV-EmbedQA-E5-v5) - for semantic search")
    print("3. NeMo Guardrails (optional) - for content safety")
    
    print("\n🔑 To get your NVIDIA API key:")
    print("1. Visit: https://build.nvidia.com/")
    print("2. Sign up or log in to your NVIDIA account")
    print("3. Go to 'API Keys' section")
    print("4. Create a new API key")
    print("5. Copy the API key")
    
    # Get API key from user
    api_key = input("\n🔑 Enter your NVIDIA API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return False
    
    if api_key == "your_nvidia_api_key_here":
        print("❌ Please enter your actual API key, not the placeholder.")
        return False
    
    # Update .env file
    try:
        # Read current .env content
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Replace placeholder API keys
        content = content.replace("your_nvidia_api_key_here", api_key)
        
        # Write back to .env
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ API key updated in .env file")
        
        # Test the configuration
        print("\n🧪 Testing NVIDIA API configuration...")
        return test_nvidia_config(api_key)
        
    except Exception as e:
        print(f"❌ Error updating .env file: {e}")
        return False

def test_nvidia_config(api_key):
    """Test NVIDIA API configuration."""
    try:
        # Set environment variables
        os.environ["NVIDIA_API_KEY"] = api_key
        os.environ["LLM_NIM_URL"] = "https://integrate.api.nvidia.com/v1"
        os.environ["EMBEDDING_NIM_URL"] = "https://integrate.api.nvidia.com/v1"
        
        # Test imports
        print("📦 Testing imports...")
        from src.api.services.llm.nim_client import NIMClient, NIMConfig
        
        # Create client
        print("🔧 Creating NIM client...")
        config = NIMConfig()
        client = NIMClient(config)
        
        print("✅ NVIDIA API configuration successful!")
        print("\n🎯 Next steps:")
        print("1. Test the API endpoint: curl -X POST http://localhost:8001/api/v1/chat -H 'Content-Type: application/json' -d '{\"message\":\"What is the stock level for SKU123?\"}'")
        print("2. Check the response for structured data and natural language")
        print("3. Verify LLM reasoning is working")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Verify your API key is correct")
        print("2. Check your internet connection")
        print("3. Ensure you have access to NVIDIA NIM services")
        return False

if __name__ == "__main__":
    success = setup_nvidia_api()
    if success:
        print("\n🎉 Setup complete! Your Warehouse Operational Assistant is ready with NVIDIA NIM integration.")
    else:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)
