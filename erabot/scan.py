"""`erabot scan` — the full agentic audit, run locally via the erabot container.

Thin launcher: it starts the in-VPC container (which holds the agentic engine),
mounts your code read-only, opens a browser file-picker, and audits on demand.
Your source never leaves your machine. Docker is required (the engine + Claude
CLI ship in the image). No key of your own needed — the first few scans run on a
free trial, metered by a one-time email signup; bring your own ANTHROPIC_API_KEY
to skip the trial.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

import typer
from rich.console import Console

console = Console()

_IMAGE = "ghcr.io/rohan3008/erabot-invpc:latest"
_API = "https://api.erabot.ai"
_CONFIG = Path.home() / ".erabot" / "config.json"


def _docker_problem() -> str | None:
    if not shutil.which("docker"):
        return ("Docker is required to run the full audit (the engine ships in a container).\n"
                "  Install Docker Desktop: https://www.docker.com/products/docker-desktop/\n"
                "  Or try the zero-install static estimate instead:  erabot estimate")
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=20, check=True)
    except Exception:
        return "Docker is installed but not running — start Docker Desktop and try again."
    return None


def _token(email: str | None) -> str:
    """Reuse a cached trial token, else mint one from a one-time email."""
    if _CONFIG.exists():
        try:
            cached = json.loads(_CONFIG.read_text()).get("token")
            if cached:
                return cached
        except Exception:  # noqa: BLE001 — corrupt cache → just re-mint
            pass
    if not email:
        console.print("[dim]The scan runs on a free trial (a few audits, on us). "
                      "One-time email — nothing else is collected.[/dim]")
        email = typer.prompt("Your email")
    body = json.dumps({"email": email}).encode()
    req = urllib.request.Request(f"{_API}/v1/onboard", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as resp:  # noqa: S310 — our own API
        tok = json.loads(resp.read())["token"]
    _CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG.write_text(json.dumps({"token": tok, "email": email}))
    return tok


def _free_port(start: int = 8787) -> int:
    for p in range(start, start + 25):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def run_scan(path: str = ".", email: str | None = None, port: int | None = None) -> None:
    problem = _docker_problem()
    if problem:
        console.print(f"[red]{problem}[/red]")
        raise typer.Exit(1)

    target = Path(path).resolve()
    if not target.exists():
        console.print(f"[red]Path not found: {target}[/red]")
        raise typer.Exit(1)

    token = _token(email)
    port = port or _free_port()

    console.print(f"Starting erabot — your code never leaves your machine. "
                  f"Mounting [bold]{target}[/bold] (read-only).")
    subprocess.run(["docker", "pull", "-q", _IMAGE], capture_output=True)  # best-effort refresh

    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    cmd = ["docker", "run", "--rm", "-p", f"{port}:8787",
           "-v", f"{target}:/repo:ro", "-v", "erabot-out:/out",
           "-e", f"ERABOT_INSTANCE_ID={token}"]
    if env_key:                                   # bring-your-own-key skips the trial
        cmd += ["-e", f"ANTHROPIC_API_KEY={env_key}"]
    cmd.append(_IMAGE)

    url = f"http://localhost:{port}"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    opened = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if not opened and "Uvicorn running" in line:
                time.sleep(1.0)
                webbrowser.open(url)
                opened = True
                console.print(f"\n[green]▸ Opening {url} — pick a folder or file, then Scan.[/green]")
                console.print("[dim]  (leave this running; Ctrl+C to stop)[/dim]\n")
        proc.wait()
    except KeyboardInterrupt:
        console.print("\nStopping erabot…")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
