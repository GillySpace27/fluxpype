"""Fluxpype Plumber: a local web GUI for the fluxpype pipeline.

Only the standard library is imported at package level; the server modules add
FastAPI/uvicorn, and nothing here imports the science stack.
"""

__all__ = ["paths", "config_io"]
