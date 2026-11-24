# -*- coding: utf-8 -*-
"""The agentg2p desktop window: ``agentg2p ui``.

A local server (server.py) serves the interface in static/ and answers its
calls with the library. ``agentg2p ui`` shows it in a native window
(pywebview); ``agentg2p ui --browser`` opens it in the default browser instead.
"""

import argparse
import sys
import threading
import webbrowser

from .server import Api, UIServer

__all__ = ["main", "start_server"]


def start_server(port: int = 0, data_dir=None) -> UIServer:
    """Starts the server on 127.0.0.1 in a background thread and returns it."""
    server = UIServer(Api(data_dir), port)
    threading.Thread(target=server.serve_forever, name="agentg2p-ui", daemon=True).start()
    return server


def _browser(server: UIServer, open_it: bool) -> int:
    print(f"agentg2p: the interface is at {server.url} (Ctrl+C to stop)", flush=True)
    if open_it:
        webbrowser.open(server.url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agentg2p ui", description="Open the agentg2p window.")
    p.add_argument("--browser", action="store_true", help="open in the default browser instead")
    p.add_argument("--no-open", action="store_true", help="with --browser: only print the address")
    p.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: any free port)")
    p.add_argument("--data-dir", help="data folder (default: $AGENTG2P_DATA, else ~/.cache/agentg2p)")
    args = p.parse_args(argv)

    server = start_server(args.port, args.data_dir)
    if args.browser:
        return _browser(server, not args.no_open)
    try:
        import webview
    except ImportError as e:
        print(f"agentg2p: the desktop window needs pywebview ({e}); opening the browser instead.",
              file=sys.stderr)
        return _browser(server, True)
    window = webview.create_window("agentg2p", server.url, width=1280, height=820,
                                   min_size=(960, 600), background_color="#FFFFFF")
    server.api.window = window
    try:
        webview.start()
    except Exception as e:  # e.g. Linux without GTK or Qt
        print(f"agentg2p: the desktop window could not start ({e}); opening the browser instead.",
              file=sys.stderr)
        server.api.window = None
        return _browser(server, True)
    server.shutdown()
    return 0
