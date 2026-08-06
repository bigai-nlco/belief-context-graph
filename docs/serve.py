#!/usr/bin/env python3
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
import webbrowser
import threading

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
host, port = "127.0.0.1", 8000
url = f"http://{host}:{port}/index.html"

print(f"BCG offline documentation: {url}")
print("Press Ctrl+C to stop.")

threading.Timer(0.6, lambda: webbrowser.open(url)).start()
ThreadingHTTPServer((host, port), SimpleHTTPRequestHandler).serve_forever()
