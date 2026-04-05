import asyncio
import threading
import queue
import logging
from typing import Dict, Any, Generator

from core.data_types import ToolResult
from core.config import AegisConfig
from utils.ollama_client import OllamaClient
from utils.openrouter_client import OpenRouterClient
from core.forensic_summary import build_phi3_prompt

logger = logging.getLogger(__name__)

async def stream_completion(prompt: str, temperature: float = 0.1, max_tokens: int = 512):
    """
    Asynchronous generator for streaming LLM completion via Client.
    """
    agent_config = AegisConfig().agent
    
    if agent_config.use_openrouter:
        client = OpenRouterClient(agent_config)
    else:
        client = OllamaClient(agent_config)
        
    q = asyncio.Queue()
    
    async def stream_callback(token: str):
        await q.put(token)
        
    async def run_gen():
        success = False
        if agent_config.use_openrouter:
            client = OpenRouterClient(agent_config)
            try:
                async with client:
                    await client.generate(
                        prompt=prompt,
                        stream_callback=stream_callback,
                        use_streaming=True
                    )
                success = True
            except Exception as e:
                logger.warning(f"OpenRouter failed: {e}. Falling back to Ollama.")
                await q.put("\n\n[OpenRouter failed. Falling back to local Ollama model...]\n\n")
                
        if not success:
            client = OllamaClient(agent_config)
            try:
                async with client:
                    await client.generate(
                        prompt=prompt,
                        stream_callback=stream_callback,
                        use_streaming=True
                    )
            except Exception as e:
                await q.put(f"__ERROR__: Both OpenRouter and Ollama failed: {str(e)}")
                
        await q.put(None)
        
    task = asyncio.create_task(run_gen())
    
    while True:
        token = await q.get()
        if token is None:
            break
        elif token.startswith("__ERROR__: "):
            error_msg = token.replace("__ERROR__: ", "")
            yield f"\n[LLM Error: {error_msg}]"
        else:
            yield token
            
    await task


def generate_verdict(
    ensemble_score: float,
    tool_results: Dict[str, ToolResult],
    verdict: str,
    config: AegisConfig = None
) -> Generator[Any, None, str]:
    """
    Generate LLM verdict using either OllamaClient or OpenRouterClient
    with streaming support. Bridges asynchronous Client with the synchronous ForensicAgent.
    """
    prompt = build_phi3_prompt(ensemble_score, tool_results, verdict)
    agent_config = config.agent if config else AegisConfig().agent
    q = queue.Queue()
    
    def async_runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def run():
                def stream_callback(token: str):
                    q.put(token)

                success = False
                if agent_config.use_openrouter:
                    client = OpenRouterClient(agent_config)
                    try:
                        async with client:
                            await client.generate(
                                prompt=prompt,
                                stream_callback=stream_callback,
                                use_streaming=True
                            )
                        success = True
                    except Exception as e:
                        logger.warning(f"OpenRouter failed: {e}. Falling back to Ollama.")
                        q.put("\n\n[OpenRouter failed. Falling back to local Ollama model...]\n\n")
                        
                if not success:
                    client = OllamaClient(agent_config)
                    try:
                        async with client:
                            await client.generate(
                                prompt=prompt,
                                stream_callback=stream_callback,
                                use_streaming=True
                            )
                    except Exception as e:
                        q.put(f"__ERROR__: Both OpenRouter and Ollama failed: {str(e)}")
                        
                q.put(None)

            loop.run_until_complete(run())
            
            # Clean up pending tasks to avoid "Task was destroyed but it is pending" error
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
        finally:
            # Only close loop after all run_until_complete are done
            loop.close()
        
    t = threading.Thread(target=async_runner)
    t.start()
    
    explanation = ""
    while True:
        try:
            token = q.get(timeout=300)
        except queue.Empty:
            logger.error("LLM generation timed out")
            break
            
        if token is None:
            break
        elif token.startswith("__ERROR__: "):
            error_msg = token.replace("__ERROR__: ", "")
            from core.agent import AgentEvent
            yield AgentEvent("llm_stream", data={"token": f"\n[LLM Error: {error_msg}]"})
            explanation += f"\n[LLM Error: {error_msg}]"
        else:
            explanation += token
            from core.agent import AgentEvent
            yield AgentEvent("llm_stream", data={"token": token})
            
    t.join()
    return explanation
