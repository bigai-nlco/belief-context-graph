"""Credential-free client mounted inside the strict CLM sandbox."""
import json
import socket
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 /tools/research.py TOOL '{\"arg\": \"value\"}'")
    request = {"name": sys.argv[1], "args": json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}}
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(150)
        sock.connect("/research-rpc/tools.sock")
        sock.sendall(json.dumps(request).encode() + b"\n")
        with sock.makefile("rb") as stream:
            response = json.loads(stream.readline(8 * 1024 * 1024))
    if not response.get("ok"):
        raise SystemExit(response.get("error", "Tool request failed"))
    result = response["result"]
    print(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
