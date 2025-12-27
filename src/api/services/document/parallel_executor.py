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
Parallel Execution Utilities for Document Processing

Provides utilities for parallelizing independent processing stages
to improve overall pipeline performance.
"""

import asyncio
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


async def execute_parallel(
    tasks: List[Tuple[str, Callable, tuple, dict]],
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute multiple independent tasks in parallel using asyncio.gather().
    
    Args:
        tasks: List of tuples (task_name, coroutine_function, args, kwargs)
        timeout: Optional timeout in seconds for all tasks
        
    Returns:
        Dictionary mapping task names to their results
        
    Example:
        results = await execute_parallel([
            ("ocr", ocr_processor.extract_text, (images,), {}),
            ("layout", layout_detector.detect_layout, (preprocessing_result,), {}),
        ])
        ocr_result = results["ocr"]
        layout_result = results["layout"]
    """
    if not tasks:
        return {}
    
    logger.info(f"Executing {len(tasks)} tasks in parallel: {[name for name, _, _, _ in tasks]}")
    start_time = datetime.now()
    
    # Create coroutines for all tasks
    coroutines = []
    task_names = []
    for task_name, func, args, kwargs in tasks:
        coroutines.append(func(*args, **kwargs))
        task_names.append(task_name)
    
    try:
        # Execute all tasks in parallel
        if timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*coroutines, return_exceptions=True),
                timeout=timeout
            )
        else:
            results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Build result dictionary
        result_dict = {}
        for i, (task_name, result) in enumerate(zip(task_names, results)):
            if isinstance(result, Exception):
                logger.error(f"Task {task_name} failed: {result}")
                result_dict[task_name] = None
            else:
                result_dict[task_name] = result
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Parallel execution completed in {elapsed:.2f}s")
        
        return result_dict
        
    except asyncio.TimeoutError:
        logger.error(f"Parallel execution timed out after {timeout}s")
        raise
    except Exception as e:
        logger.error(f"Parallel execution failed: {e}")
        raise


async def batch_api_calls(
    api_func: Callable,
    items: List[Any],
    batch_size: int = 5,
    timeout_per_item: Optional[float] = None,
) -> List[Any]:
    """
    Batch API calls to process multiple items concurrently.
    
    Args:
        api_func: Async function to call for each item
        items: List of items to process
        batch_size: Number of concurrent API calls per batch
        timeout_per_item: Optional timeout per item
        
    Returns:
        List of results in the same order as items
        
    Example:
        embeddings = await batch_api_calls(
            embedding_service.generate_embedding,
            text_chunks,
            batch_size=10
        )
    """
    if not items:
        return []
    
    logger.info(f"Batching {len(items)} items with batch_size={batch_size}")
    results = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
        
        # Create coroutines for this batch
        coroutines = [api_func(item) for item in batch]
        
        try:
            if timeout_per_item:
                # Apply timeout per item
                batch_results = await asyncio.gather(
                    *[asyncio.wait_for(coro, timeout=timeout_per_item) for coro in coroutines],
                    return_exceptions=True
                )
            else:
                batch_results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch item {i + j} failed: {result}")
                    results.append(None)
                else:
                    results.append(result)
                    
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            # Add None for failed batch items
            results.extend([None] * len(batch))
    
    logger.info(f"Batch processing completed: {len([r for r in results if r is not None])}/{len(items)} successful")
    return results

