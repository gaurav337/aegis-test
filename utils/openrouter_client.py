import asyncio
import json
import logging
from typing import Optional, Callable, Dict, Any, Union

import httpx

from core.config import AgentConfig

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    Async OpenRouter API client for cloud LLM generations.
    Supports stream tracking and reasoning tokens.
    """
    
    def __init__(self, config: AgentConfig):
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = getattr(config, 'openrouter_api_key', '')
        self.model_name = getattr(config, 'openrouter_model', 'qwen/qwen3.6-plus:free')
        self.timeout = getattr(config, 'llm_timeout', 120)
        self.temperature = float(getattr(config, 'llm_temperature', 0.1))
        
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
    async def __aenter__(self) -> 'OpenRouterClient':
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
        
    async def close(self) -> None:
        async with self._client_lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self._client = None
                
    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Aegis-X Pipeline"
                }
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout, connect=10.0),
                    headers=headers
                )
            return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Any]]] = None,
        use_streaming: bool = False
    ) -> str:
        """Execute OpenRouter chat completion."""
        if not self.api_key:
            error_msg = "OpenRouter API Key not found in .env"
            logger.error(error_msg)
            if stream_callback:
                stream_callback(f"[Error: {error_msg}]")
            return error_msg

        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
            
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": use_streaming
        }

        try:
            if use_streaming:
                accumulated = []
                async with client.stream("POST", self.endpoint, json=payload, timeout=self.timeout) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        logger.error(f"OpenRouter Error {response.status_code}: {err_text}")
                        raise Exception(f"OpenRouter returned {response.status_code}")
                        
                    async for line in response.aiter_lines():
                        if not line.strip() or not line.startswith("data: "):
                            continue
                        
                        data_str = line[6:].strip() # remove "data: "
                        if data_str == "[DONE]":
                            break
                            
                        try:
                            chunk = json.loads(data_str)
                            
                            # Handle reasoning tokens (if provided)
                            # E.g. usage info in openrouter arrives in the chunk
                            
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated.append(content)
                                if stream_callback:
                                    stream_callback(content)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in chunk: {data_str}")
                            
                return "".join(accumulated)
            else:
                response = await client.post(self.endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
        except httpx.ConnectError as e:
            logger.error(f"OpenRouter connection failed: {e}")
            raise e
        except httpx.TimeoutException as e:
            logger.error(f"OpenRouter timeout: {e}")
            raise e
        except Exception as e:
            logger.error(f"OpenRouter unexpected error: {e}")
            raise e
