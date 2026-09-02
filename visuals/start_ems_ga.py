"""Reliable launcher for the EMS + GA live decision viewer."""
from __future__ import annotations

import functools
import http.server
import os
import socket
import socketserver
import threading
import time
import webbrowser
from pathlib import Path

DIR = Path(__file__).resolve().parent
PAGE = "ems_ga_live.html"
PORTS = (8766, 8767, 8768, 8771, 8890)


def free_port() -> int:
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        if args and str(args[0]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    os.chdir(DIR)
    html = DIR / PAGE
    if not html.is_file():
        print(f"ERROR: missing {html}")
        input("Press Enter to exit...")
        return 1

    port = free_port()
    handler = functools.partial(Handler, directory=str(DIR))
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        print(f"ERROR: could not bind port {port}: {exc}")
        input("Press Enter to exit...")
        return 1

    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/{PAGE}"

    print()
    print("=" * 52)
    print("  EMS + Genetic Algorithm — Live Decisions")
    print("=" * 52)
    print()
    print(f"  Folder : {DIR}")
    print(f"  URL    : {url}")
    print()
    print("  Opening your browser...")
    print("  KEEP THIS WINDOW OPEN while you use the viewer.")
    print("  Press Ctrl+C or close this window to stop.")
    print()

    time.sleep(0.5)
    if not webbrowser.open(url, new=1):
        print("  Auto-open failed. Paste this into Chrome or Edge:")
        print(f"  {url}")
        print()
    for name in ("windows-default", "chrome", "msedge"):
        try:
            webbrowser.get(name).open(url)
            break
        except Exception:
            continue

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping...")
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
