"""TensorBoard logger for MORL baselines."""

import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """A logger class that wraps TensorBoard functionality for MORL experiments.

    This class provides a similar interface to wandb for easy migration.
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        mode: Optional[str] = "online",
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the TensorBoard logger.

        Args:
            project_name: Name of the project (used in log directory)
            experiment_name: Name of the experiment
            entity: Optional entity (used in log directory)
            group: Optional group for grouping runs
            mode: Mode of logging (not used for TensorBoard, kept for compatibility)
            config: Configuration dictionary to save
            seed: Random seed for the run
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.entity = entity
        self.group = group
        self.mode = mode
        self.config = config or {}
        self.seed = seed

        # Create a unique run name
        timestamp = int(time.time())
        self.run_name = f"{experiment_name}__{seed}__{timestamp}"

        # Build log directory path
        log_dir_parts = [project_name, experiment_name]
        if entity:
            log_dir_parts.insert(0, entity)
        if group:
            log_dir_parts.append(group)
        log_dir_parts.append(self.run_name)

        self.log_dir = os.path.join("runs", *log_dir_parts)

        # Create the SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Store config as text for reference
        if config:
            config_text = "\n".join([f"{k}: {v}" for k, v in config.items()])
            self.writer.add_text("config", config_text)

    def log(self, data: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """Log metrics to TensorBoard.

        Args:
            data: Dictionary of metrics to log
            step: Global step for the metrics
            commit: Whether to commit the metrics (kept for wandb compatibility)
        """
        if step is None:
            # Try to get step from data, default to 0
            step = data.get("global_step", 0)

        for key, value in data.items():
            if key == "global_step":
                continue

            # Handle different value types
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(key, value, global_step=step)
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    self.writer.add_scalar(key, value.item(), global_step=step)
                elif value.ndim == 1:
                    # Log each element of 1D array as separate scalar
                    for i, v in enumerate(value):
                        self.writer.add_scalar(f"{key}_{i}", float(v), global_step=step)
            elif isinstance(value, Table):
                # Handle wandb.Table equivalent
                self._log_table(key, value, step)

    def _log_table(self, name: str, table: "Table", step: int):
        """Log a table to TensorBoard.

        Args:
            name: Name of the table
            table: Table object
            step: Global step
        """
        # Convert table data to markdown format for TensorBoard
        headers = table.columns
        data = table.data

        # Create markdown table
        markdown_lines = []
        markdown_lines.append("| " + " | ".join(headers) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in data:
            row_str = " | ".join([str(x) for x in row])
            markdown_lines.append(f"| {row_str} |")

        markdown_table = "\n".join(markdown_lines)
        self.writer.add_text(name, markdown_table, global_step=step)

    def add_text(self, tag: str, text: str, step: int = 0):
        """Add text to TensorBoard.

        Args:
            tag: Tag for the text
            text: Text content
            step: Global step
        """
        self.writer.add_text(tag, text, global_step=step)

    def add_image(self, tag: str, img: np.ndarray, step: int = 0):
        """Add an image to TensorBoard.

        Args:
            tag: Tag for the image
            img: Image as numpy array (HWC format)
            step: Global step
        """
        self.writer.add_image(tag, img, global_step=step, dataformats="HWC")

    def add_figure(self, tag: str, figure, step: int = 0):
        """Add a matplotlib figure to TensorBoard.

        Args:
            tag: Tag for the figure
            figure: Matplotlib figure
            step: Global step
        """
        self.writer.add_figure(tag, figure, global_step=step)

    def add_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Add hyperparameters and associated metrics.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of metrics
        """
        self.writer.add_hparams(hparams, metrics)

    def finish(self):
        """Close the TensorBoard writer."""
        self.writer.close()

    @property
    def summary(self):
        """Return a mock summary object for compatibility with wandb.run.summary.

        This is a simplified version that stores values in a dictionary.
        """
        return TensorBoardSummary(self)


class TensorBoardSummary:
    """Mock summary object for wandb.run.summary compatibility."""

    def __init__(self, logger: TensorBoardLogger):
        self.logger = logger
        self._data = {}

    def __setitem__(self, key: str, value: Any):
        """Set a summary value.

        Args:
            key: Key for the summary
            value: Value to store
        """
        self._data[key] = value
        # Also log to TensorBoard
        if isinstance(value, (int, float, np.number)):
            # Use a large step value to indicate this is a final summary
            self.logger.writer.add_scalar(key, value, global_step=999999999)

    def __getitem__(self, key: str) -> Any:
        """Get a summary value.

        Args:
            key: Key for the summary

        Returns:
            The stored value
        """
        return self._data.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the summary.

        Args:
            key: Key to check

        Returns:
            True if key exists
        """
        return key in self._data


# Global variable to track the current logger (similar to wandb's global state)
_current_logger: Optional[TensorBoardLogger] = None


def init(
    project: str = "MORL-Baselines",
    name: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    monitor_gym: bool = True,
    save_code: bool = True,
    group: Optional[str] = None,
    mode: Optional[str] = "online",
    dir: Optional[str] = None,
    id: Optional[str] = None,
    resume: Optional[str] = "never",
    **kwargs,
) -> TensorBoardLogger:
    """Initialize TensorBoard logging.

    This function mimics wandb.init() for easy migration.

    Args:
        project: Project name
        name: Run name
        entity: Entity name
        config: Configuration dictionary
        monitor_gym: Whether to monitor gym (not used for TensorBoard)
        save_code: Whether to save code (not used for TensorBoard)
        group: Group name
        mode: Logging mode
        dir: Directory for logs
        id: Run ID (not used for TensorBoard)
        resume: Resume option (not used for TensorBoard)
        **kwargs: Additional arguments

    Returns:
        TensorBoardLogger instance
    """
    global _current_logger

    # Extract experiment name from the name parameter
    experiment_name = name or "experiment"

    # Extract seed from config if available
    seed = config.get("seed") if config else None

    _current_logger = TensorBoardLogger(
        project_name=project,
        experiment_name=experiment_name,
        entity=entity,
        group=group,
        mode=mode,
        config=config,
        seed=seed,
    )

    return _current_logger


def log(data: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
    """Log data to TensorBoard.

    Args:
        data: Dictionary of metrics to log
        step: Global step
        commit: Whether to commit (kept for wandb compatibility)
    """
    if _current_logger is not None:
        _current_logger.log(data, step, commit)


def finish():
    """Finish the current run."""
    global _current_logger
    if _current_logger is not None:
        _current_logger.finish()
        _current_logger = None


def define_metric(metric_name: str, step_metric: str):
    """Define a metric with a custom step.

    For TensorBoard, this is a no-op since steps are handled automatically.

    Args:
        metric_name: Name of the metric
        step_metric: Step metric name
    """
    pass  # TensorBoard handles steps automatically


# Mock config object for wandb.config compatibility
class Config:
    """Mock config object for wandb.config compatibility."""

    def __init__(self):
        self._data = {}

    def __setitem__(self, key: str, value: Any):
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def update(self, data: Dict[str, Any]):
        self._data.update(data)


config = Config()


class Table:
    """Mock Table class for wandb.Table compatibility."""

    def __init__(self, columns: List[str], data: Optional[List[List[Any]]] = None):
        self.columns = columns
        self.data = data or []

    def add_data(self, *args):
        """Add a row of data to the table.

        Args:
            *args: Values for the row
        """
        if len(args) != len(self.columns):
            raise ValueError(f"Expected {len(self.columns)} values, got {len(args)}")
        self.data.append(list(args))


class Run:
    """Mock Run class for wandb.run compatibility."""

    def __init__(self, logger: TensorBoardLogger):
        self.logger = logger
        self.summary = TensorBoardSummary(logger)

    @property
    def dir(self) -> str:
        """Return the log directory."""
        return self.logger.log_dir


def get_current_logger() -> Optional[TensorBoardLogger]:
    """Get the current logger instance.

    Returns:
        Current TensorBoardLogger instance or None
    """
    return _current_logger


# For backward compatibility with code that accesses wandb.run
class _RunAccessor:
    """Accessor for the current run."""

    def __init__(self):
        self._run = None

    def __call__(self) -> Optional[Run]:
        if _current_logger is not None and self._run is None:
            self._run = Run(_current_logger)
        elif _current_logger is not None:
            # Update the run's logger if it changed
            self._run.logger = _current_logger
            self._run.summary = TensorBoardSummary(_current_logger)
        return self._run

    def __getattr__(self, name: str):
        run = self()
        if run is not None:
            return getattr(run, name)
        return None


run = _RunAccessor()


# Sweep-related stubs for compatibility
def sweep(sweep_config: Dict, entity: Optional[str] = None, project: str = "MORL-Baselines") -> str:
    """Mock sweep function.

    For TensorBoard, sweeps need to be run manually with different hyperparameters.
    This function returns a mock sweep ID.

    Args:
        sweep_config: Sweep configuration
        entity: Entity name
        project: Project name

    Returns:
        Mock sweep ID
    """
    print("Warning: wandb.sweep() is not fully supported with TensorBoard.")
    print("For hyperparameter sweeps, run multiple experiments with different configurations manually.")
    return "mock_sweep_id"


def agent(sweep_id: str, function: callable, count: int = 1):
    """Mock agent function for sweeps.

    This is a no-op for TensorBoard since sweeps are not automated.

    Args:
        sweep_id: Sweep ID
        function: Function to run
        count: Number of runs
    """
    print("Warning: wandb.agent() is not supported with TensorBoard.")
    print("Run your sweep configurations manually for TensorBoard logging.")
