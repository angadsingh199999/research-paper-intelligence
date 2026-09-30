import sys
import time
import subprocess
import urllib.request
import json

# Start streamlit on port 8599
cmd = [
    sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
    "--server.port", "8599",
    "--server.headless", "true",
]

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

time.sleep(3)

try:
    # Check health endpoint
    try:
        with urllib.request.urlopen("http://localhost:8599/_stcore/health", timeout=3) as resp:
            print("Health status:", resp.status, resp.read().decode())
    except Exception as e:
        print("Health check error:", e)

    # Check root page
    try:
        with urllib.request.urlopen("http://localhost:8599/", timeout=3) as resp:
            content = resp.read().decode()
            print("Root page status:", resp.status, "Length:", len(content))
            print("Root title in HTML:", "<title>" in content)
    except Exception as e:
        print("Root page error:", e)

finally:
    proc.terminate()
    proc.wait(timeout=3)
    stdout, _ = proc.communicate()
    print("Streamlit output:")
    print(stdout[:1000])
