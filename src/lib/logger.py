import logging
from logging import Logger
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_tracebacks


def get_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> Logger:
    """
    Enhanced logger for NN projects.
    Supports both console (Rich) and file output.
    """
    install_rich_tracebacks(show_locals=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 2. Console Handler (Rich)
    console = Console(stderr=True)
    rich_handler = RichHandler(
        console=console,
        level=level,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=False,
        show_path=False,
    )
    logger.addHandler(rich_handler)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    # 3. File Handler (Plain text - better for grep/searching)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create a filename based on the logger name (e.g., 'train_run.log')
        file_handler = logging.FileHandler(log_path / f"{name}.log")
        file_handler.setLevel(level)

        # Standard format for file logs (easier to parse later)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Example usage for a PINN project:
if __name__ == "__main__":
    # In your main script:
    logger = get_logger(name="Schrodinger_PINN_v1")

    logger.info("Starting training on [bold cyan]GPU:0[/bold cyan]...")

    # Simulate training log
    epoch = 1000
    loss = 1.23e-4
    logger.info(f"Epoch: {epoch} | Total Loss: [green]{loss:.2f}[/green]")
