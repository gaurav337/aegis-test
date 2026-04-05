import os
import tempfile
import pickle
import subprocess
import time
import logging

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult

logger = logging.getLogger(__name__)

class SubprocessToolProxy(BaseForensicTool):
    """
    Acts as a proxy for tools that must run in an isolated environment (e.g. .venv_gpu).
    Serializes TrackedFace inputs, runs subprocess_worker.py, and deserializes ToolResult.
    """
    def __init__(self, tool_name: str, python_exec: str = None):
        super().__init__()
        self._tool_name = tool_name
        self.requires_gpu = True
        
        # Find python exec
        if python_exec is None:
            venv_gpu = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.venv_gpu'))
            if os.name == 'nt':
                self.python_exec = os.path.join(venv_gpu, 'Scripts', 'python.exe')
            else:
                self.python_exec = os.path.join(venv_gpu, 'bin', 'python')
        else:
            self.python_exec = python_exec
            
        self.worker_script = os.path.abspath(os.path.join(os.path.dirname(__file__), 'subprocess_worker.py'))

    @property
    def tool_name(self) -> str:
        return self._tool_name
        
    def setup(self):
        # Delegate setup to worker during execute for isolation
        pass

    def health_check(self) -> ToolResult:
        if not os.path.exists(self.python_exec):
            return ToolResult(self.tool_name, success=False, error=True, error_msg="Python executable not found")
        # Ensure worker script exists
        if not os.path.exists(self.worker_script):
            return ToolResult(self.tool_name, success=False, error=True, error_msg="Worker script not found")
        return ToolResult(self.tool_name, success=True)

    def _run_inference(self, input_data: dict) -> ToolResult:
        start = time.perf_counter()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_in:
            in_path = temp_in.name
            
        out_path = in_path + ".out"
        
        try:
            with open(in_path, 'wb') as f:
                # GPU tools only need face crops, first_frame, media_path, and context.
                # Strip frames_30fps — it can be hundreds of full-resolution video frames
                # (potentially 100MB+), causing massive pickle files that time out.
                slim_input = {k: v for k, v in input_data.items() if k != 'frames_30fps'}
                pickle.dump(slim_input, f)
            
            # Build env: inherit current environment + ensure AEGIS_MODEL_DIR is always set
            env = os.environ.copy()
            if 'AEGIS_MODEL_DIR' not in env:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                env['AEGIS_MODEL_DIR'] = os.path.join(project_root, 'models')
                
            cmd = [self.python_exec, self.worker_script, self.tool_name, in_path]
            
            # Capture stderr for debugging; subprocess logs are routed there
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            
            # Always forward subprocess stderr to our logger so model loading is visible
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    logger.info(f"[{self.tool_name} worker] {line}")
            
            if result.returncode != 0:
                logger.error(f"{self.tool_name} subprocess worker failed. Stderr:\n{result.stderr}\nStdout:\n{result.stdout}")
                return ToolResult(
                    tool_name=self.tool_name,
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    details={"stderr": result.stderr},
                    error=True,
                    error_msg=f"Subprocess worker failed with return code {result.returncode}",
                    execution_time=time.perf_counter() - start,
                    evidence_summary="Subprocess error"
                )
                
            if not os.path.exists(out_path):
                return ToolResult(
                    tool_name=self.tool_name,
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    details={"stdout": result.stdout, "stderr": result.stderr},
                    error=True,
                    error_msg="Subprocess did not generate an output configuration file",
                    execution_time=time.perf_counter() - start,
                    evidence_summary="Subprocess error"
                )
                
            with open(out_path, 'rb') as f:
                worker_result = pickle.load(f)
                
            if isinstance(worker_result, Exception):
                logger.error(f"{self.tool_name} worker raised Exception: {worker_result}")
                return ToolResult(
                    tool_name=self.tool_name,
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    details={},
                    error=True,
                    error_msg=str(worker_result),
                    execution_time=time.perf_counter() - start,
                    evidence_summary="Worker Exception"
                )
                
            return worker_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"{self.tool_name} proxy execution timed out after 120s")
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg="Subprocess timed out after 120s",
                execution_time=time.perf_counter() - start,
                evidence_summary="Timeout"
            )
        except Exception as e:
            logger.error(f"{self.tool_name} proxy execution failed: {e}", exc_info=True)
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg=str(e),
                execution_time=time.perf_counter() - start,
                evidence_summary="Proxy Exception"
            )
            
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)
