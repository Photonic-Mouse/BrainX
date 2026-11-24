import os
import sys
import time
import socket
import urllib.request
import json
import subprocess
import webbrowser
import signal

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
HEALTH_URL = f"{URL}/api/health"

CHECKPOINTS = [
    ("Baseline ResNet18", r"checkpoints\resnet18_baseline.pth"),
    ("HE ResNet18", r"checkpoints\resnet18_he.pth"),
    ("CLAHE ResNet18", r"checkpoints\resnet18_clahe.pth")
]

REQUIRED_PROJECT_FILES = [
    "run_app.py",
    os.path.join("src", "models", "resnet.py"),
    os.path.join("src", "inference", "pipeline.py")
]


def print_banner():
    print("=" * 68)
    print("      BRAINX: MULTI-MODEL BRAIN MRI ANALYSIS PLATFORM")
    print("          College Demonstration & Research Launcher")
    print("=" * 68)
    print()


def check_project_root():
    """Verify that execution is occurring from the BrainX project root."""
    print("[1/5] Checking project directory structure...")
    missing = [f for f in REQUIRED_PROJECT_FILES if not os.path.exists(f)]
    if missing:
        print(f"\n[ERROR] Not in BrainX project root or files missing:")
        for m in missing:
            print(f"  - Missing: {m}")
        print("\nPlease run START_BRAINX.bat directly from the BrainX root directory.")
        return False
    print("      -> Project root verified: " + os.getcwd())
    return True


def check_checkpoints():
    """Verify that all three trained model checkpoints exist."""
    print("\n[2/5] Verifying model checkpoints...")
    all_ok = True
    for name, path in CHECKPOINTS:
        if not os.path.exists(path):
            print(f"      [MISSING] {name} checkpoint not found at: {path}")
            all_ok = False
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"      -> {name}: {os.path.basename(path)} ({size_mb:.1f} MB) [OK]")
    
    if not all_ok:
        print("\n[ERROR] Model checkpoints are missing. Cannot continue without weights.")
        return False
    return True


def check_cuda_environment():
    """Display CUDA GPU acceleration status and hardware info."""
    print("\n[3/5] Inspecting compute device & deep learning environment...")
    try:
        import torch
        print(f"      -> PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"      -> Compute Hardware: {device_name} ({vram_gb:.1f} GB VRAM)")
            print(f"      -> CUDA Acceleration: ACTIVE (Device cuda:0)")
        else:
            print("      -> CUDA Acceleration: Not detected. Using CPU fallback mode.")
    except Exception as e:
        print(f"      [WARN] Could not inspect PyTorch device info: {e}")


def check_port_and_health():
    """
    Check whether port 8000 is open.
    Returns:
      'brainx_running': Port is open and /api/health confirms BrainX is already running.
      'port_occupied': Port is open, but did not respond with BrainX health info.
      'port_free': Port is available to bind.
    """
    print(f"\n[4/5] Checking port {PORT} availability...")
    # 1. First check if BrainX is already running
    try:
        req = urllib.request.Request(HEALTH_URL, headers={"User-Agent": "BrainX-Launcher"})
        with urllib.request.urlopen(req, timeout=1.5) as res:
            if res.status == 200:
                body = json.loads(res.read().decode())
                if body.get("status") == "healthy" and "models_loaded" in body:
                    print("      -> BrainX is already running on port 8000.")
                    return "brainx_running"
    except Exception:
        pass

    # 2. Try binding with SO_REUSEADDR to see if port is free for Uvicorn
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.close()
        print(f"      -> Port {PORT} is available.")
        return "port_free"
    except Exception:
        # Brief retry in case of lingering connection tear-down
        time.sleep(2)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.close()
            print(f"      -> Port {PORT} is available.")
            return "port_free"
        except Exception:
            return "port_occupied"


def wait_for_backend_ready(timeout_seconds=35):
    """Polls the backend health endpoint until it responds with HTTP 200."""
    print("      Waiting for backend to initialize models and start web server...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        try:
            req = urllib.request.Request(HEALTH_URL, headers={"User-Agent": "BrainX-Launcher"})
            with urllib.request.urlopen(req, timeout=1.5) as res:
                if res.status == 200:
                    data = json.loads(res.read().decode())
                    if data.get("status") == "healthy":
                        elapsed = time.time() - start_time
                        print(f"      -> Backend healthy & ready! (initialized in {elapsed:.1f}s)")
                        return True
        except Exception:
            pass
        time.sleep(0.6)
        sys.stdout.write(".")
        sys.stdout.flush()

    print("\n[ERROR] Backend did not respond within timeout period.")
    return False


def main():
    print_banner()

    # 1. Project Root Check
    if not check_project_root():
        sys.exit(1)

    # 2. Checkpoints Check
    if not check_checkpoints():
        sys.exit(1)

    # 3. Hardware / CUDA status
    check_cuda_environment()

    # 4. Port & Running Server Check
    status = check_port_and_health()

    if status == "brainx_running":
        print("\n" + "=" * 68)
        print("  [NOTICE] BrainX server is ALREADY RUNNING and healthy on port 8000!")
        print(f"  Opening the active dashboard in your browser: {URL}")
        print("=" * 68)
        try:
            webbrowser.open(URL)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
        
        print("\nOptions:")
        print("  - To continue using BrainX, leave this window or close it.")
        print("  - To shutdown the running server, run STOP_BRAINX.bat.")
        input("\nPress [Enter] to exit launcher...")
        sys.exit(0)

    elif status == "port_occupied":
        print(f"\n[ERROR] Port {PORT} is currently occupied by another program.")
        print(f"Please close any application using port {PORT} (or run STOP_BRAINX.bat) and try again.")
        sys.exit(1)

    # 5. Start Backend Server
    print(f"\n[5/5] Launching BrainX FastAPI Server (Uvicorn)...")
    server_cmd = [sys.executable, "run_app.py"]

    # Start server as subprocess
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Wait for server to become healthy
    ready = wait_for_backend_ready()
    if not ready:
        print("\n[ERROR] Failed to start BrainX server. Diagnostic logs:")
        server_proc.terminate()
        stdout, _ = server_proc.communicate()
        print(stdout)
        sys.exit(1)

    # 6. Open Browser
    print(f"\n[LAUNCH] Opening browser to: {URL}")
    try:
        webbrowser.open(URL)
    except Exception as e:
        print(f"Could not launch browser automatically: {e}")

    print("\n" + "=" * 68)
    print("  BRAINX IS LIVE & READY FOR YOUR DEMONSTRATION!")
    print(f"  URL: {URL}")
    print()
    print("  Controls:")
    print("    - Press [Ctrl + C] in this window to cleanly shutdown BrainX.")
    print("    - Or run STOP_BRAINX.bat from another window.")
    print("=" * 68)
    print()

    # Stream server logs and handle clean shutdown
    try:
        for line in server_proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Stopping BrainX server process gracefully...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
            print("[SHUTDOWN] BrainX server stopped successfully.")
        except subprocess.TimeoutExpired:
            server_proc.kill()
            print("[SHUTDOWN] Server forced closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 68)
        print("  [FATAL ERROR] An unexpected exception occurred during launch:")
        print("=" * 68)
        traceback.print_exc()
        sys.exit(1)
