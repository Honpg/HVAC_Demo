"""
Startup helper: Start Uvicorn server in background when Streamlit runs
"""

import subprocess
import threading
import time
import os
import sys
import socket
from typing import Optional

# Global reference to the Uvicorn process
_uvicorn_process: Optional[subprocess.Popen] = None
_uvicorn_thread: Optional[threading.Thread] = None


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_uvicorn_background(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> bool:
    """
    Start Uvicorn server in background subprocess.
    
    Args:
        host: Server host (default: 127.0.0.1)
        port: Server port (default: 8000)
        reload: Enable auto-reload (default: False for production)
    
    Returns:
        True if server started successfully, False if port already in use
    """
    global _uvicorn_process
    
    # Check if port is already in use
    if is_port_in_use(port):
        print(f"⚠️ Port {port} already in use. Using existing server.")
        return True
    
    try:
        # Build command
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.api:app",
            "--host", host,
            "--port", str(port),
        ]
        
        if reload:
            cmd.append("--reload")
        
        # Start process
        _uvicorn_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Wait a bit for server to start
        time.sleep(2)
        
        if _uvicorn_process.poll() is None:  # Process is still running
            print(f"✅ Uvicorn server started on http://{host}:{port}")
            return True
        else:
            print(f"❌ Failed to start Uvicorn server")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Uvicorn: {e}")
        return False


def stop_uvicorn_background() -> None:
    """Stop the background Uvicorn server."""
    global _uvicorn_process
    
    if _uvicorn_process is not None:
        try:
            if os.name == 'nt':
                # Windows
                os.system(f"taskkill /F /PID {_uvicorn_process.pid}")
            else:
                # Linux/Mac
                _uvicorn_process.terminate()
                _uvicorn_process.wait(timeout=5)
            print(f"✅ Uvicorn server stopped")
        except Exception as e:
            print(f"⚠️ Error stopping Uvicorn: {e}")
        finally:
            _uvicorn_process = None


def ensure_uvicorn_running(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> bool:
    """
    Ensure Uvicorn is running. Start if not already running.
    Safe to call multiple times.
    
    Returns:
        True if Uvicorn is running, False otherwise
    """
    global _uvicorn_process
    
    # If process exists and still running
    if _uvicorn_process is not None and _uvicorn_process.poll() is None:
        return True
    
    # If process died, clean up
    if _uvicorn_process is not None:
        _uvicorn_process = None
    
    # Try to start
    return start_uvicorn_background(host=host, port=port, reload=reload)
