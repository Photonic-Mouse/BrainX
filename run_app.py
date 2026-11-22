import os
import sys
import time
import threading
import webbrowser
import uvicorn

def launch_browser():
    """Wait for server initialization and open browser automatically."""
    time.sleep(1.8)
    try:
        webbrowser.open("http://127.0.0.1:8000")
        print("[Launcher] Browser opened to http://127.0.0.1:8000")
    except Exception as e:
        print(f"[Launcher] Could not launch browser automatically: {e}")

if __name__ == "__main__":
    print("=" * 65)
    print("  BrainX: Interactive Multi-Model MRI Analysis Application")
    print("  Loading ResNet18 Checkpoints & Starting Uvicorn Web Server...")
    print("  URL: http://127.0.0.1:8000")
    print("=" * 65)
    
    # Launch browser in a background thread
    threading.Thread(target=launch_browser, daemon=True).start()
    
    # Run uvicorn server
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
