#!/usr/bin/env python3
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
host, port = "127.0.0.1", 8000
url = f"http://{host}:{port}/index.html"

print(f"BCG offline documentation: {url}")
print("Press Ctrl+C to stop.")

threading.Timer(0.6, lambda: webbrowser.open(url)).start()
ThreadingHTTPServer((host, port), SimpleHTTPRequestHandler).serve_forever()
