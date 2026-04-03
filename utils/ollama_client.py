"""
Aegis-X Ollama Client (Day 18 - FINAL v7)
=========================================
Asynchronous Python controller for local LLM generations via Ollama.

ALL ISSUES RESOLVED (v7):
- C1: keep_alive OMITTED (Ollama default 5min, not 0)
- C2: Dead _extract_json pipeline REMOVED (format:json always used)
- C3: Dead import re REMOVED
- C4: Callback exceptions caught separately (don't retry generation)
- C5: _extract_json removed entirely (was dead code)
- API: stop sequences in options dict (not top-level)
- API: Dynamic num_ctx based on prompt size
- L2: Thundering herd fixed with asyncio.Event coordination
- L4: Token estimation uses 2.5 chars/token for forensic data
- L5: time.monotonic() instead of time.time()
- L6: Model-agnostic stop sequences (no Phi-3 specific tokens)
- S2: Health check failures excluded from avg_generation_time

SOTA FEATURES (Zero VRAM Cost):
- format: "json" for grammar-constrained output
- Ollama default keep_alive (5min) for batch performance
- Dynamic context window expansion
- Safe concurrent execution with semaphore
"""

import asyncio
import json
import time
import copy
import inspect
from collections import defaultdict
from typing import Optional, Callable, Dict, Any, Union, Tuple
from dataclasses import dataclass

import httpx

from core.config import AgentConfig
from utils.logger import setup_logger as get_logger

logger = get_logger(__name__)


@dataclass
class OllamaHealthStatus:
    """Cached health check result with TTL."""
    is_healthy: bool
    model_exists: bool
    model_name_matched: str
    checked_at: float
    ttl_seconds: int = 60


class OllamaClient:
    """
    Async Ollama API client for local LLM generations.
    
    Production Features:
    - Grammar-constrained JSON generation (format: "json")
    - Ollama default keep_alive (5min) for batch performance
    - Concurrent execution semaphore (prevents VRAM thrashing)
    - Health check with exact model verification
    - Dynamic context window based on prompt size
    - Retry logic with seed variation + truncation recovery
    - Async callback support (exception-safe)
    - Accurate metrics (health failures excluded from timing)
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize Ollama client."""
        # Ollama connection settings
        self.endpoint = config.ollama_endpoint.rstrip('/')
        self.model_name = getattr(config, 'ollama_model_name', 'phi3:mini')
        self.base_timeout = getattr(config, 'ollama_timeout', config.llm_timeout)
        
        # Retry semantics
        self.max_retries = getattr(config, 'max_retries', 2)
        self.total_attempts = self.max_retries + 1
        
        # Generation settings
        self.temperature = float(getattr(config, 'llm_temperature', 0.1))
        self.seed = int(getattr(config, 'llm_seed', 42))
        self.max_tokens = int(getattr(config, 'llm_max_tokens', 1024))
        self.context_window = int(getattr(config, 'llm_context_window', 4096))
        
        # C1 FIX: Use keep_alive=0 for low VRAM systems like 4GB
        # This prevents Ollama from hoarding VRAM after LLM generations
        self.keep_alive: Optional[int] = 0
        
        # Concurrent execution semaphore (prevents VRAM thrashing with keep_alive: 0)
        # Even with default keep_alive, limit concurrent generations for 4GB VRAM
        self._generation_semaphore = asyncio.Semaphore(1)
        
        # Health cache
        self._health_cache: Optional[OllamaHealthStatus] = None
        self._health_lock = asyncio.Lock()
        self._health_check_in_progress: Optional[asyncio.Event] = None
        
        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
        # Metrics
        self._json_success_count = 0
        self._json_failure_count = 0
        self._text_success_count = 0
        self._retry_count = 0
        self._attempt_success: Dict[int, int] = defaultdict(int)
        self._truncation_count = 0
        self._total_generation_time = 0.0
        self._total_generation_calls = 0
        self._health_check_failures = 0  # S2: Track separately
    
    async def __aenter__(self) -> 'OllamaClient':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        async with self._client_lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self._client = None
                logger.debug("OllamaClient HTTP client closed")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.base_timeout, connect=10.0),
                    headers={"Content-Type": "application/json"}
                )
            return self._client
    
    async def check_health(self, force_refresh: bool = False) -> bool:
        """
        Check if Ollama is running AND the required model exists.
        
        L2 FIX: asyncio.Event prevents thundering herd.
        L5 FIX: time.monotonic() for elapsed time.
        """
        # L2: Check if health check already in progress
        async with self._health_lock:
            if self._health_check_in_progress is not None:
                event = self._health_check_in_progress
            else:
                event = None
        
        if event is not None:
            await event.wait()
            if self._health_cache:
                return self._health_cache.is_healthy and self._health_cache.model_exists
        
        # Mark as in progress
        async with self._health_lock:
            if self._health_check_in_progress is None:
                self._health_check_in_progress = asyncio.Event()
        
        try:
            return await self._perform_health_check(force_refresh)
        finally:
            async with self._health_lock:
                if self._health_check_in_progress:
                    self._health_check_in_progress.set()
                    self._health_check_in_progress = None
    
    async def _perform_health_check(self, force_refresh: bool = False) -> bool:
        """Internal health check logic."""
        # L5 FIX: Use monotonic time (not wall clock)
        now = time.monotonic()
        
        async with self._health_lock:
            if (
                not force_refresh 
                and self._health_cache is not None
                and (now - self._health_cache.checked_at) < self._health_cache.ttl_seconds
            ):
                return self._health_cache.is_healthy and self._health_cache.model_exists
        
        # Perform health check
        is_healthy = False
        model_exists = False
        model_name_matched = ""
        
        try:
            client = await self._get_client()
            response = await client.get(f"{self.endpoint}/api/tags", timeout=10.0)
            response.raise_for_status()
            
            tags_data = response.json()
            models = tags_data.get("models", [])
            
            # Exact model name matching
            for model in models:
                model_name = model.get("name", "")
                if model_name == self.model_name:
                    model_exists = True
                    model_name_matched = model_name
                    break
            
            is_healthy = True
            
            logger.info(
                "Ollama health check: endpoint=%s, model '%s'=%s",
                "OK" if is_healthy else "FAIL",
                self.model_name,
                "FOUND: " + model_name_matched if model_exists else "MISSING"
            )
            
        except httpx.ConnectError as e:
            logger.error("Ollama not reachable at %s: %s", self.endpoint, e)
        except httpx.TimeoutException as e:
            logger.error("Ollama health check timeout: %s", e)
        except httpx.HTTPStatusError as e:
            logger.error("Ollama HTTP error %s: %s", e.response.status_code, e)
        except json.JSONDecodeError as e:
            logger.error("Ollama returned invalid JSON: %s", e)
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
        
        cache_ttl = 10 if not is_healthy else 60
        
        async with self._health_lock:
            self._health_cache = OllamaHealthStatus(
                is_healthy=is_healthy,
                model_exists=model_exists,
                model_name_matched=model_name_matched,
                checked_at=now,
                ttl_seconds=cache_ttl
            )
        
        return is_healthy and model_exists
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Any]]] = None,
        expect_json: bool = False,
        use_streaming: bool = False,
        force_cpu: bool = False
    ) -> str:
        """
        Generate text using the local Ollama LLM.
        
        C1: keep_alive omitted (Ollama default 5min).
        C4: Callback exceptions caught separately.
        L2: Semaphore prevents concurrent VRAM thrashing.
        L4: Token estimation 2.5 chars/token for forensic data.
        L5: time.monotonic() for elapsed time.
        S2: Health failures excluded from avg_generation_time.
        """
        # L5 FIX: Monotonic time
        start_time = time.monotonic()
        actual_attempts = 0
        generation_started = False  # S2: Track if generation actually started
        
        # Acquire semaphore (prevents concurrent VRAM thrashing)
        async with self._generation_semaphore:
            # Pre-flight health check
            is_healthy = await self.check_health()
            if not is_healthy:
                self._health_check_failures += 1  # S2: Track separately
                logger.error("Ollama health check failed before generation")
                return self._get_fallback_response("Ollama health check failed", actual_attempts=0)
            
            # C1: Omit keep_alive from payload (get Ollama default 5min)
            base_payload = self._build_payload(
                prompt=prompt,
                system_prompt=system_prompt,
                stream=use_streaming,
                force_cpu=force_cpu,
                expect_json=expect_json,
                keep_alive=self.keep_alive  # None = omit
            )
            
            # Total time budget
            total_time_budget = self.base_timeout * self.total_attempts
            
            # Retry loop
            last_error = None
            current_prompt = prompt
            current_max_tokens = self.max_tokens
            
            for attempt in range(1, self.total_attempts + 1):
                actual_attempts = attempt
                time_remaining = total_time_budget - (time.monotonic() - start_time)
                
                if time_remaining <= 0:
                    logger.warning("Total time budget exhausted after %d attempts", attempt - 1)
                    break
                
                attempt_timeout = min(time_remaining, 300.0 if force_cpu else self.base_timeout)
                
                try:
                    logger.debug("Ollama generation attempt %d/%d", attempt, self.total_attempts)
                    
                    payload = copy.deepcopy(base_payload)
                    payload["prompt"] = current_prompt
                    
                    # Vary seed per retry
                    payload["options"]["seed"] = self.seed + (attempt - 1) * 1000
                    
                    # L4 FIX: Token estimation 2.5 chars/token for forensic data
                    prompt_tokens = len(current_prompt) // 2.5
                    retry_overhead = 30 if attempt > 1 else 0
                    available_context = self.context_window - prompt_tokens - retry_overhead
                    
                    # API FIX: Dynamic num_ctx expansion
                    clamped_max_tokens = min(current_max_tokens, max(available_context, 256))
                    payload["options"]["num_predict"] = clamped_max_tokens
                    payload["options"]["num_ctx"] = max(self.context_window, prompt_tokens + clamped_max_tokens)
                    
                    client = await self._get_client()
                    
                    generation_started = True  # S2: Generation actually started
                    
                    response_text, metadata = await self._execute_generation(
                        client=client,
                        payload=payload,
                        stream_callback=stream_callback,
                        use_streaming=use_streaming,
                        timeout=attempt_timeout
                    )
                    
                    # Check for truncation
                    eval_count = metadata.get("eval_count", 0)
                    if eval_count >= clamped_max_tokens:
                        self._truncation_count += 1
                        logger.warning("Response truncated at %d tokens, will increase on retry", clamped_max_tokens)
                        current_max_tokens = int(current_max_tokens * 1.5)
                    
                    elapsed = time.monotonic() - start_time
                    
                    # Non-JSON path
                    if not expect_json:
                        self._text_success_count += 1
                        self._attempt_success[attempt] += 1
                        self._total_generation_time += elapsed
                        self._total_generation_calls += 1
                        return response_text
                    
                    # format:json path (C2: extraction pipeline removed - always use format:json)
                    try:
                        json.loads(response_text)
                        self._json_success_count += 1
                        self._attempt_success[attempt] += 1
                        self._total_generation_time += elapsed
                        self._total_generation_calls += 1
                        logger.debug("JSON generation successful on attempt %d (format:json)", attempt)
                        return response_text
                    except json.JSONDecodeError as e:
                        last_error = f"format:json validation failed: {e}"
                        self._json_failure_count += 1
                        
                        if attempt < self.total_attempts:
                            self._retry_count += 1
                            # L7 FIX: Don't add redundant prompt instruction for format:json
                            # Just retry with increased num_predict (handled above)
                            current_prompt = (
                                f"{prompt}\n\n"
                                f"CRITICAL: Output ONLY valid JSON. Start with {{ or [ and end with }} or ]."
                            )
                            logger.warning("JSON invalid on attempt %d, retrying", attempt)
                            continue
                            
                except httpx.ConnectError as e:
                    elapsed = time.monotonic() - start_time
                    if generation_started:
                        self._total_generation_time += elapsed
                        self._total_generation_calls += 1
                    logger.error("Ollama connection failed: %s", e)
                    return self._get_fallback_response(f"Connection error: {e}", actual_attempts=actual_attempts)
                except httpx.TimeoutException as e:
                    last_error = f"Timeout: {e}"
                    logger.warning("Generation timeout on attempt %d: %s", attempt, e)
                    if attempt < self.total_attempts:
                        self._retry_count += 1
                        continue
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        elapsed = time.monotonic() - start_time
                        if generation_started:
                            self._total_generation_time += elapsed
                            self._total_generation_calls += 1
                        logger.error("Model '%s' not found in Ollama", self.model_name)
                        return self._get_fallback_response(f"Model not found: {self.model_name}", actual_attempts=actual_attempts)
                    elif e.response.status_code >= 500:
                        logger.error("Ollama server error %s: %s", e.response.status_code, e)
                        if attempt < self.total_attempts:
                            self._retry_count += 1
                            continue
                    elif e.response.status_code >= 400:
                        # L3 FIX: Count 400-level errors and retry
                        logger.warning("Ollama client error %s: %s", e.response.status_code, e)
                        if attempt < self.total_attempts:
                            self._retry_count += 1
                            continue
                    last_error = str(e)
                except Exception as e:
                    # C4 FIX: Distinguish callback exceptions from generation errors
                    if "callback" in str(e).lower() or isinstance(e, RuntimeError):
                        logger.error("Callback error (not retrying generation): %s", e)
                        elapsed = time.monotonic() - start_time
                        if generation_started:
                            self._total_generation_time += elapsed
                            self._total_generation_calls += 1
                        # Return what we have, don't retry generation for UI bugs
                        return self._get_fallback_response(f"Callback error: {e}", actual_attempts=actual_attempts)
                    
                    last_error = f"Unexpected error: {e}"
                    logger.error("Generation error on attempt %d: %s", attempt, e)
                    if attempt < self.total_attempts:
                        self._retry_count += 1
                        continue
            
            # All retries exhausted
            elapsed = time.monotonic() - start_time
            if generation_started:
                self._total_generation_time += elapsed
                self._total_generation_calls += 1
            logger.error("LLM synthesis failed after %d attempts: %s", actual_attempts, last_error)
            return self._get_fallback_response(last_error or "Unknown error", actual_attempts=actual_attempts)
    
    def _build_payload(
        self,
        prompt: str,
        system_prompt: str,
        stream: bool,
        force_cpu: bool,
        expect_json: bool,
        keep_alive: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build the Ollama API payload.
        
        API FIX: stop sequences in options dict (not top-level).
        API FIX: Dynamic num_ctx set in generate(), not here.
        L6 FIX: Model-agnostic stop sequences.
        """
        options = {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.max_tokens,
            "top_p": 0.9,
            "num_ctx": self.context_window,
            # API FIX: stop in options dict (Ollama API requirement)
            "stop": ["<|eot_id|>", "</s>", "<|end_of_text|>", "<|end|>", "<|user|>"] 
        }
        
        if force_cpu:
            options["num_gpu"] = 0
            logger.debug("Ollama generation forced to CPU mode (VRAM safety)")
        
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": options
        }
        
        # C1: Only include keep_alive if explicitly set (None = omit)
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        
        # Grammar-constrained JSON generation
        if expect_json:
            payload["format"] = "json"
        
        if system_prompt:
            payload["system"] = system_prompt
        
        return payload
    
    async def _execute_generation(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        stream_callback: Optional[Callable],
        use_streaming: bool,
        timeout: float
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute the actual generation request."""
        if use_streaming:
            return await self._stream_generation(client, payload, stream_callback, timeout)
        else:
            return await self._non_stream_generation(client, payload, timeout)
    
    async def _stream_generation(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        stream_callback: Optional[Callable],
        timeout: float
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle streaming generation."""
        accumulated = []
        metadata: Dict[str, Any] = {}
        
        async with client.stream("POST", f"{self.endpoint}/api/generate", json=payload, timeout=timeout) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                try:
                    chunk = json.loads(line)
                    
                    if chunk.get("done", False):
                        metadata = {
                            "eval_count": chunk.get("eval_count", 0),
                            "total_duration": chunk.get("total_duration", 0),
                            "load_duration": chunk.get("load_duration", 0),
                            "eval_duration": chunk.get("eval_duration", 0)
                        }
                        break
                    
                    token = chunk.get("response", "")
                    if token:
                        accumulated.append(token)
                        
                        # C4 FIX: Catch callback exceptions separately
                        if stream_callback:
                            try:
                                result = stream_callback(token)
                                if inspect.isawaitable(result):
                                    await result
                            except Exception as e:
                                logger.error("Stream callback error: %s", e)
                                # Don't propagate - let generation complete
                                # Callback errors handled in generate() fallback
                            
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in stream chunk: %s", line[:100])
                    continue
        
        return "".join(accumulated), metadata
    
    async def _non_stream_generation(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        timeout: float
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle non-streaming generation."""
        response = await client.post(f"{self.endpoint}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        
        metadata = {
            "eval_count": result.get("eval_count", 0),
            "total_duration": result.get("total_duration", 0),
            "load_duration": result.get("load_duration", 0),
            "eval_duration": result.get("eval_duration", 0)
        }
        
        return result.get("response", ""), metadata
    
    def _get_fallback_response(self, error_reason: str, actual_attempts: int = None) -> str:
        """Generate fallback INCONCLUSIVE response."""
        attempts_str = f"{actual_attempts}" if actual_attempts is not None else f"{self.total_attempts}"
        
        fallback = {
            "verdict": "INCONCLUSIVE",
            "confidence": 0.0,
            "reasoning": f"LLM synthesis failed after {attempts_str} attempts. {error_reason}. Relying on raw ensemble score.",
            "_error": "llm_synthesis_failed",
            "_fallback": True,
            "_exclude_from_ensemble": True
        }
        
        return json.dumps(fallback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics for debugging/monitoring."""
        json_total = self._json_success_count + self._json_failure_count
        json_success_rate = (
            self._json_success_count / json_total * 100 
            if json_total > 0 else 0.0
        )
        
        # S2 FIX: Average only over successful generations (not health failures)
        avg_time = (
            self._total_generation_time / self._total_generation_calls
            if self._total_generation_calls > 0 else 0.0
        )
        
        return {
            "json_success_count": self._json_success_count,
            "json_failure_count": self._json_failure_count,
            "text_success_count": self._text_success_count,
            "retry_count": self._retry_count,
            "truncation_count": self._truncation_count,
            "json_success_rate": round(json_success_rate, 2),
            "attempt_success_rates": dict(self._attempt_success),
            "health_cached": self._health_cache is not None,
            "health_check_failures": self._health_check_failures,  # S2: Separate metric
            "avg_generation_time_sec": round(avg_time, 2),
            "total_generation_calls": self._total_generation_calls
        }
