"""Vercel Python entrypoint for the LedgerLens backend.

Vercel's @vercel/python runtime serves the module-level ASGI `app`. The
vercel.json rewrite sends every path here, and the FastAPI app's own routes
(/chat, /resume, /health) receive the original request path unchanged — so no
root_path is needed.

Deploy this directory (`backend/`) as its own Vercel project (root directory =
backend). `requirements.txt` beside this folder is installed automatically.
"""
import os
import sys

# Ensure the project root (backend/) is importable so `app` resolves whether or
# not the platform already put it on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402,F401  (ASGI app served by Vercel)
