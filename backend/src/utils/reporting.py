import logging
from pathlib import Path
from typing import Any


def analyze_optuna_study(study: Any, output_dir: str, _run_id: str) -> None:
    """Analyze an Optuna study and save diagnostic plots to output_dir.

    This function treats Optuna plotting as optional. If the plotting
    libraries (plotly/kaleido) are not available or an error occurs while
    generating images, the function logs the error and returns without
    raising so training runs can complete successfully.
    """
    logger = logging.getLogger("nftool")

    try:
        vis: Any = __import__("optuna.visualization", fromlist=["*"])
        pio: Any = __import__("plotly.io", fromlist=["*"])
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        logger.info("Optuna plotting skipped: %s", exc)
        return

    out = Path(output_dir)
    try:
        try:
            if hasattr(pio, "kaleido") and hasattr(pio.kaleido, "scope"):
                pio.kaleido.scope.chromium_path = "/usr/bin/chromium"
        except (AttributeError, RuntimeError, OSError, ValueError):
            logger.info("Optuna plotting: unable to set chromium path")

        # 1. Optimization History
        fig = vis.plot_optimization_history(study)
        try:
            fig.write_image(str(out / "optuna_optimization_history.png"))
        except (RuntimeError, OSError, ValueError):
            fig.write_html(str(out / "optuna_optimization_history.html"))

        # 2. Parameter Importances
        if len(study.trials) > 1:
            fig = vis.plot_param_importances(study)
            try:
                fig.write_image(str(out / "optuna_param_importances.png"))
            except (RuntimeError, OSError, ValueError):
                fig.write_html(str(out / "optuna_param_importances.html"))

        # 3. Parallel Coordinate
        fig = vis.plot_parallel_coordinate(study)
        try:
            fig.write_image(str(out / "optuna_parallel_coordinate.png"))
        except (RuntimeError, OSError, ValueError):
            fig.write_html(str(out / "optuna_parallel_coordinate.html"))
    except (
        ValueError,
        ImportError,
        OSError,
        RuntimeError,
    ) as exc:  # guard against runtime plotting errors (kaleido/plotly issues)
        logger.info("Optuna plotting failed: %s", exc)
        return


def generate_regression_plots(y_true: Any, y_pred: Any, output_dir: str) -> None:
    """Generate and save standard regression diagnostic plots."""
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    # 1. Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color="#3b82f6")
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
        label="Ideal",
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual (Regression)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(str(out / "predicted_vs_actual.png"))
    plt.close()

    # 2. Residual Distribution
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color="#3b82f6", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.grid(alpha=0.3)
    plt.savefig(str(out / "residual_distribution.png"))
    plt.close()
