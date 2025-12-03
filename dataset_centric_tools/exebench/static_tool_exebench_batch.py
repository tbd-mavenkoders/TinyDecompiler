"""
Parallelized Gemini API processing for ExeBench dataset
Optimized for high-throughput batch processing with strict rate limiting
"""

import asyncio
import aiohttp
import yaml
import json
import tempfile
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta
import math

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing modules
from utils.llm_interface import create_llm_interface, clean_llm_output
from utils.compile import Compiler, OptimizationLevel
from utils.clean_errors import ErrorNormalizer


@dataclass
class ProcessingStats:
    """Track processing statistics"""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    skipped_empty: int = 0
    start_time: datetime = None
    api_calls_made: int = 0
    
    def success_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.successful / self.total_processed
    
    def elapsed_time(self) -> timedelta:
        if self.start_time:
            return datetime.now() - self.start_time
        return timedelta(0)
    
    def estimated_remaining(self, total_items: int) -> timedelta:
        if self.total_processed == 0:
            return timedelta(0)
        
        elapsed = self.elapsed_time()
        rate = self.total_processed / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        remaining_items = total_items - self.total_processed
        
        if rate > 0:
            return timedelta(seconds=remaining_items / rate)
        return timedelta(0)


class StrictRateLimiter:
    """
    Strict rate limiter for ExeBench with lower daily limits
    90 requests/minute, 10,000 requests/day
    """
    
    def __init__(self, max_requests_per_minute: int = 80, max_requests_per_day: int = 9500):
        self.max_requests_per_minute = max_requests_per_minute  # Safety buffer
        self.max_requests_per_day = max_requests_per_day
        self.request_times = []
        self.daily_request_count = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request with daily limit checking"""
        async with self.lock:
            now = time.time()
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Reset daily counter if it's a new day
            if current_date > self.daily_reset_time:
                self.daily_request_count = 0
                self.daily_reset_time = current_date
                print(f"📅 Daily request counter reset. Used {self.daily_request_count} requests yesterday.")
            
            # Check daily limit
            if self.daily_request_count >= self.max_requests_per_day:
                print(f"⚠️  Daily limit of {self.max_requests_per_day} requests reached!")
                print(f"⏰ Waiting until tomorrow to continue...")
                # Calculate time until tomorrow
                tomorrow = current_date + timedelta(days=1)
                sleep_seconds = (tomorrow - datetime.now()).total_seconds()
                await asyncio.sleep(sleep_seconds)
                self.daily_request_count = 0
                self.daily_reset_time = tomorrow
            
            # Remove requests older than 1 minute for per-minute limit
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            # Check per-minute limit
            if len(self.request_times) >= self.max_requests_per_minute:
                oldest_request = min(self.request_times)
                sleep_time = 60 - (now - oldest_request) + 0.1  # Add small buffer
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Clean up again after waiting
                    now = time.time()
                    self.request_times = [t for t in self.request_times if now - t < 60]
            
            # Record this request
            self.request_times.append(now)
            self.daily_request_count += 1


class ExeBenchBatchProcessor:
    """Batch processor for ExeBench dataset with async Gemini API calls"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.load_config()
        self.setup_paths()
        self.compiler = Compiler()
        self.rate_limiter = StrictRateLimiter()
        self.stats = ProcessingStats()
        
        # Conservative concurrency for ExeBench (larger dataset, stricter limits)
        self.max_concurrent_requests = 15  # Reduced from 20 for ExeBench
        self.max_concurrent_compiles = 8   
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.compile_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_compiles)
        
        # Progress tracking
        self.progress_lock = asyncio.Lock()
        
        # Batch processing for memory efficiency with large dataset
        self.batch_size = 100  # Process in smaller batches
        
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
    
    def setup_paths(self):
        """Setup input and output paths"""
        self.corpus_root = Path(self.config["paths"]["exebench_corpus_root"])
        self.output_dir = Path(self.config["paths"]["exebench_static_repair_path"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_initial_prompt(self, c_code: str, language: str) -> str:
        """Generate initial prompt for LLM"""
        initial_prompt = self.config["prompts"]["system_prompt"]
        return f"{initial_prompt}\n\n```Language:{language}\n{c_code}\n```"
    
    async def make_llm_request(self, prompt: str) -> str:
        """Make async LLM request with strict rate limiting"""
        await self.rate_limiter.acquire()
        
        # Create a new LLM interface for each request
        llm_interface = create_llm_interface(
            provider=self.config["llm"]["gemini_provider"],
            model_name=self.config["llm"]["gemini_model_name"],
            api_key=self.config["llm"]["gemini_api_key"]
        )
        
        # Run the synchronous LLM call in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, llm_interface.generate, prompt)
        
        async with self.progress_lock:
            self.stats.api_calls_made += 1
        
        return clean_llm_output(result)
    
    async def compile_code_async(self, code: str, language: str) -> Tuple[bool, str]:
        """Async wrapper for compilation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.compile_executor, 
            self._compile_code_sync, 
            code, 
            language
        )
    
    def _compile_code_sync(self, code: str, language: str) -> Tuple[bool, str]:
        """Synchronous compilation in thread pool"""
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir) / "code.c"
            output_file = Path(temp_dir) / "code.out"
            
            with open(code_file, "w") as f:
                f.write(code)
            
            status, message = self.compiler.compile_source(
                source_file_path=str(code_file),
                output_file_path=str(output_file),
                opt=OptimizationLevel.O0,
                is_cpp=(language == "cpp")
            )
            
            return status, message
    
    async def process_single_item(self, data: Dict, max_iterations: int = 10) -> Dict:
        """Process a single test case with async operations"""
        async with self.semaphore:  # Limit concurrent requests
            try:
                original_code = data["ghidra_pseudo"].strip()
                language = data["language"]
                func_name = data.get("func_name", "unknown")
                
                # Skip empty functions
                if len(original_code) == 0:
                    print(f"⏭️  {func_name}: Skipping empty function")
                    data["optimized_func"] = original_code
                    data["optimization_status"] = False
                    data["skip_reason"] = "empty_function"
                    async with self.progress_lock:
                        self.stats.skipped_empty += 1
                    return data
                
                # Check if original code compiles
                status, message = await self.compile_code_async(original_code, language)
                
                if status:
                    print(f"✓ {func_name}: Original code compiles, no optimization needed")
                    data["optimized_func"] = original_code
                    data["optimization_status"] = True
                    return data
                
                # Initial LLM optimization
                print(f"⚡ {func_name}: Starting optimization...")
                initial_prompt = self.get_initial_prompt(original_code, language)
                optimized_code = await self.make_llm_request(initial_prompt)
                
                # Check if initially optimized code compiles
                status, message = await self.compile_code_async(optimized_code, language)
                
                if status:
                    print(f"✓ {func_name}: Optimized successfully in 1 iteration")
                    data["optimized_func"] = optimized_code
                    data["optimization_status"] = True
                    return data
                
                # Iterative repair loop
                error_normalizer = ErrorNormalizer()
                
                for iteration in range(max_iterations - 1):
                    print(f"🔄 {func_name}: Iteration {iteration + 2}")
                    
                    # Generate repair prompt
                    error_prompt = error_normalizer.format_for_llm(message)
                    repair_prompt = (
                        f"{self.config['prompts']['compilation_error']}\n\n"
                        f"```Language:{language}\n{optimized_code}\n```\n\n"
                        f"Compilation Errors:\n{error_prompt}\n\n"
                        f"Please provide the corrected C code."
                    )
                    
                    # Get LLM repair
                    optimized_code = await self.make_llm_request(repair_prompt)
                    
                    # Check compilation
                    status, message = await self.compile_code_async(optimized_code, language)
                    
                    if status:
                        print(f"✓ {func_name}: Optimized successfully in {iteration + 2} iterations")
                        data["optimized_func"] = optimized_code
                        data["optimization_status"] = True
                        return data
                
                # Max iterations reached
                print(f"✗ {func_name}: Failed after {max_iterations} iterations")
                data["optimized_func"] = optimized_code
                data["optimization_status"] = False
                data["failure_reason"] = "max_iterations_reached"
                return data
                
            except Exception as e:
                print(f"✗ {func_name}: Error during processing: {str(e)}")
                data["optimized_func"] = data.get("ghidra_pseudo", "")
                data["optimization_status"] = False
                data["failure_reason"] = f"exception: {str(e)}"
                return data
    
    async def update_progress(self, data: Dict, total_items: int):
        """Update and display progress"""
        async with self.progress_lock:
            self.stats.total_processed += 1
            if data["optimization_status"]:
                self.stats.successful += 1
            else:
                self.stats.failed += 1
            
            # Print progress every 25 items for ExeBench
            if self.stats.total_processed % 25 == 0:
                elapsed = self.stats.elapsed_time()
                remaining = self.stats.estimated_remaining(total_items)
                
                print(f"\n📊 Progress: {self.stats.total_processed}/{total_items} "
                      f"({100*self.stats.total_processed/total_items:.1f}%)")
                print(f"✓ Success: {self.stats.successful} | ✗ Failed: {self.stats.failed} | "
                      f"⏭️  Skipped: {self.stats.skipped_empty}")
                print(f"📈 Success Rate: {100*self.stats.success_rate():.1f}%")
                print(f"⏱️  Elapsed: {elapsed} | Estimated remaining: {remaining}")
                print(f"🔗 API calls: {self.stats.api_calls_made} | Daily used: {self.rate_limiter.daily_request_count}")
    
    async def process_batch(self, dataset: List[Dict], max_iterations: int, batch_name: str = "") -> List[Dict]:
        """Process a batch of items concurrently"""
        print(f"🚀 Processing batch{batch_name}: {len(dataset)} items...")
        
        # Create tasks for all items in this batch
        tasks = [self.process_single_item(data.copy(), max_iterations) for data in dataset]
        
        # Process with progress updates
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            await self.update_progress(result, len(dataset))
        
        return results
    
    async def process_json_file(self, json_file_path: Path, max_iterations: int) -> None:
        """Process a single JSON file in batches"""
        print(f"📁 Loading ExeBench dataset from {json_file_path.name}...")
        
        # Load dataset
        with open(json_file_path, "r") as f:
            dataset = json.load(f)
        
        print(f"📊 Loaded {len(dataset)} test cases")
        print(f"🔄 Processing in batches of {self.batch_size} for memory efficiency")
        
        # Process in batches to manage memory with large dataset
        all_results = []
        num_batches = math.ceil(len(dataset) / self.batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(dataset))
            batch_data = dataset[start_idx:end_idx]
            
            batch_name = f" {batch_idx + 1}/{num_batches}"
            print(f"\n{'='*50}")
            print(f"Processing batch {batch_idx + 1}/{num_batches} (items {start_idx + 1}-{end_idx})")
            print(f"{'='*50}")
            
            batch_results = await self.process_batch(batch_data, max_iterations, batch_name)
            all_results.extend(batch_results)
            
            # Save intermediate results after each batch
            output_file_path = self.output_dir / f"optimized_{json_file_path.name}"
            with open(output_file_path, "w") as f:
                json.dump(all_results, f, indent=2)
            
            print(f"💾 Intermediate save: {len(all_results)} results saved")
            
            # Brief pause between batches to avoid overwhelming the API
            if batch_idx < num_batches - 1:
                print("⏸️  Brief pause between batches...")
                await asyncio.sleep(2)
        
        print(f"\n💾 Final results saved to {output_file_path}")
        
        # Final statistics
        elapsed = self.stats.elapsed_time()
        print(f"\n📈 Final Statistics for {json_file_path.name}:")
        print(f"Total processed: {self.stats.total_processed}")
        print(f"Successful: {self.stats.successful}")
        print(f"Failed: {self.stats.failed}")
        print(f"Skipped (empty): {self.stats.skipped_empty}")
        print(f"Success rate: {100*self.stats.success_rate():.2f}%")
        print(f"Total time: {elapsed}")
        print(f"API calls made: {self.stats.api_calls_made}")
        print(f"Average time per item: {elapsed.total_seconds()/self.stats.total_processed:.2f}s")
        print(f"Daily requests used: {self.rate_limiter.daily_request_count}/10,000")
    
    async def process_all_files(self, max_iterations: int = None):
        """Process all JSON files in the corpus directory"""
        if max_iterations is None:
            max_iterations = self.config["static_repair"]["max_iterations"]
        
        json_files = list(self.corpus_root.glob("*.json"))
        print(f"🔍 Found {len(json_files)} JSON files to process")
        
        # Estimate total items and time
        total_estimated_items = 0
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    total_estimated_items += len(data)
            except:
                pass
        
        print(f"📊 Estimated total items: ~{total_estimated_items}")
        print(f"⏱️  Estimated time (optimized): ~{total_estimated_items/70:.0f} minutes")
        print(f"🔗 Estimated API calls: ~{total_estimated_items * 2} calls")
        
        for json_file in json_files:
            print(f"\n{'='*80}")
            print(f"Processing: {json_file.name}")
            print(f"{'='*80}")
            
            # Reset stats for each file
            self.stats = ProcessingStats()
            self.stats.start_time = datetime.now()
            
            await self.process_json_file(json_file, max_iterations)
        
        # Cleanup
        self.compile_executor.shutdown(wait=True)


async def main():
    """Main async function"""
    # Config path
    config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    
    # Create processor
    processor = ExeBenchBatchProcessor(config_path)
    
    print(f"🎯 ExeBench Batch Processor v1.0")
    print(f"🚀 Optimized for 4,200+ test cases with strict rate limiting")
    print(f"📡 Rate limits: 80 req/min, 9,500 req/day")
    print(f"⚙️  Concurrency: {processor.max_concurrent_requests} requests, {processor.max_concurrent_compiles} compiles")
    print(f"📦 Batch size: {processor.batch_size} items per batch\n")
    
    # Process all files
    await processor.process_all_files()


if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        print("📊 Partial results have been saved to output directory")
        print("🔄 You can resume processing by running the script again")
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()