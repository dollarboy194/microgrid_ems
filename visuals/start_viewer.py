"""
Reliable launcher for the Tamale microgrid 3D viewer.
Starts a local HTTP server, opens the browser, and stays running until you close it.
"""
from __future__ import annotations

import functools
import http.server
import os
import socket
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

DIR = Path(__file__).resolve().parent
PORT_CANDIDATES = (8765, 8766, 8767, 8770, 8888, 9000)


def find_free_port() -> int:
    for port in PORT_CANDIDATES:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    # Last resort: OS-assigned free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Keep console readable; only print errors
        if args and str(args[0]).startswith(("4", "5")):
            super().log_message(format, *args)

    def end_headers(self) -> None:
        # Allow modules to load cleanly
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def main() -> int:
    os.chdir(DIR)
    html = DIR / "microgrid_3d.html"
    three = DIR / "vendor" / "three.module.js"
    if not html.is_file():
        print(f"ERROR: missing {html}")
        input("Press Enter to exit...")
        return 1
    if not three.is_file():
        print(f"ERROR: missing {three}")
        print("The vendor folder is required for the 3D engine.")
        input("Press Enter to exit...")
        return 1

    port = find_free_port()
    handler = functools.partial(QuietHandler, directory=str(DIR))
    # Allow address reuse so restarts work
    socketserver.TCPServer.allow_reuse_address = True

    try:
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        print(f"ERROR: could not bind port {port}: {exc}")
        input("Press Enter to exit...")
        return 1

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/microgrid_3d.html"
    print()
    print("=" * 52)
    print("  Tamale Microgrid — 3D Day Cycle Viewer")
    print("=" * 52)
    print()
    print(f"  Folder : {DIR}")
    print(f"  URL    : {url}")
    print()
    print("  Opening your browser now...")
    print("  Keep THIS window open while you use the viewer.")
    print("  Press Ctrl+C or close this window to stop.")
    print()

    # Give the server a moment, then open browser
    time.sleep(0.4)
    opened = webbrowser.open(url, new=1)
    if not opened:
        print("  Browser auto-open failed. Copy-paste this URL into Chrome/Edge:")
        print(f"  {url}")
        print()

    # Also try common browsers if default failed silently
    for browser_name in ("windows-default", "chrome", "msedge", "firefox"):
        try:
            b = webbrowser.get(browser_name)
            b.open(url)
            break
        except Exception:
            continue

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping server...")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nFATAL: {exc}")
        input("Press Enter to exit...")
        raise SystemExit(1)
