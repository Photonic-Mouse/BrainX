import os
import subprocess
import time

def stop_server():
    print("=" * 60)
    print("  BRAINX: CLEAN SHUTDOWN UTILITY")
    print("=" * 60)
    print()
    
    found = False
    try:
        out = subprocess.check_output("netstat -ano", shell=True, text=True)
        pids = set()
        for line in out.splitlines():
            if ":8000" in line and "LISTENING" in line:
                parts = line.split()
                if len(parts) >= 5:
                    pids.add(parts[-1])
        
        if pids:
            found = True
            for pid in pids:
                print(f"Terminating BrainX server process (PID: {pid})...")
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
            print("\n[SUCCESS] BrainX server on port 8000 stopped successfully.")
        else:
            print("[INFO] No active BrainX server found listening on port 8000.")
    except Exception as e:
        print(f"[INFO] Server already stopped or not running: {e}")

    print()
    print("=" * 60)

if __name__ == "__main__":
    stop_server()
