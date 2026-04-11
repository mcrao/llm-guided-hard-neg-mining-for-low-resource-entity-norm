"""Centralised logging using Rich for readable, coloured output."""

import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a Rich-formatted logger."""
    if not logging.root.handlers:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
        )
    return logging.getLogger(name)
