from __future__ import annotations
import asyncio
import infra.utils as utils
import os
from datetime import datetime 
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================================
#               CONTROL THE ROBOT
# =========================================================================
timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

num_seconds = 60
# logfile = os.path.join("logs", "wheel_tick", timestamp, "right.csv")
logfile = os.path.join("logs", "wheel_tick", "left_backward.csv")


# Robot runs this program
def getRobotPrograms(robot):
    coro = utils.Calibrator(robot, num_seconds, logfile).calibrate()
    return [coro]



# =========================================================================
#                CONSIDER MOVING TO A DIFFERENT FILE
#                Analyze the log - post calibration
# =========================================================================




@dataclass
class WinningHypothesis:
    i: int
    cx: int
    cy: int
    cz: int
    sign: int
    score: float
    mean_rt: float
    std_rt: float


def _candidate_offsets(values: np.ndarray, radius: int, step: int) -> np.ndarray:
    """Build a local search grid around the channel median."""
    c0 = int(np.median(values))
    lo = max(0, c0 - radius)
    hi = min(255, c0 + radius)
    return np.arange(lo, hi + 1, step, dtype=np.int16)


def _linear_residual_sq(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    values:  (T,)
    centers: (K,)
    returns: (K, T) squared residuals
    """
    d = values[None, :].astype(np.float32) - centers[:, None].astype(np.float32)
    return d * d


def _wrap_residual_sq_u8(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Wrap-aware squared residual for uint8-like values modulo 256.

    d_wrap(x, c) = ((x - c + 128) mod 256) - 128
    returns squared residuals with shape (K, T)
    """
    x = values[None, :].astype(np.int16)
    c = centers[:, None].astype(np.int16)
    d = ((x - c + 128) % 256) - 128
    d = d.astype(np.float32)
    return d * d


def find_adjacent_accel_hypotheses(
    csv_path: str | Path,
    *,
    offset_radius: int = 24,
    offset_step: int = 2,
    residual_mode: Literal["linear", "wrap_u8"] = "wrap_u8",
    top_k_print: int = 20,
    show_plot: bool = True,
    save_plot_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, WinningHypothesis]:
    """
    Search a CSV logfile for the best adjacent triple [i, i+1, i+2] matching
    a constant-magnitude 3-axis accelerometer model.

    CSV format
    ----------
    col 0   : timestamp
    col 1.. : raw sensor bytes in [0, 255]

    Hypothesis
    ----------
    (i, cx, cy, cz, sign)

    where:
      - i selects the adjacent triple [i, i+1, i+2] in payload-byte indexing
      - cx, cy, cz are additive sensor offsets
      - sign is included for bookkeeping, but is NOT identifiable under the
        score below because sign disappears after squaring

    Score
    -----
    For each sample t:
        rt = (xt - cx)^2 + (yt - cy)^2 + (zt - cz)^2
    or the wrap-aware modular analogue if residual_mode="wrap_u8"

    score = std(rt) / mean(rt)

    Returns
    -------
    results_df:
        One row per tested hypothesis, sorted by (i, cx, cy, cz, sign)
    best_per_i_df:
        Lowest-score hypothesis for each i
    winner:
        Overall best hypothesis
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 4:
        raise ValueError("Need at least 1 timestamp column + 3 payload columns.")

    payload = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    n_samples, n_payload = payload.shape
    if n_payload < 3:
        raise ValueError("Need at least 3 payload bytes.")

    if residual_mode == "linear":
        residual_fn = _linear_residual_sq
    elif residual_mode == "wrap_u8":
        residual_fn = _wrap_residual_sq_u8
    else:
        raise ValueError(f"Unknown residual_mode={residual_mode!r}")

    rows: list[dict] = []

    # Adjacent triples [i, i+1, i+2]
    for i in range(n_payload - 2):
        x = payload[:, i]
        y = payload[:, i + 1]
        z = payload[:, i + 2]

        cx_vals = _candidate_offsets(x, offset_radius, offset_step)
        cy_vals = _candidate_offsets(y, offset_radius, offset_step)
        cz_vals = _candidate_offsets(z, offset_radius, offset_step)

        # Precompute per-axis squared residual tables
        # Shapes: (Kx,T), (Ky,T), (Kz,T)
        x_sq = residual_fn(x, cx_vals)
        y_sq = residual_fn(y, cy_vals)
        z_sq = residual_fn(z, cz_vals)

        # Vectorized combination:
        # for each cx,cy pair, add all cz hypotheses at once
        for ix, cx in enumerate(cx_vals):
            x_part = x_sq[ix]  # (T,)
            for iy, cy in enumerate(cy_vals):
                xy_part = x_part + y_sq[iy]  # (T,)

                # all cz at once -> (Kz, T)
                rt_all = z_sq + xy_part[None, :]

                mean_rt = rt_all.mean(axis=1)
                std_rt = rt_all.std(axis=1)

                with np.errstate(divide="ignore", invalid="ignore"):
                    scores = std_rt / mean_rt
                scores = np.where(mean_rt > 0, scores, np.inf)

                for iz, cz in enumerate(cz_vals):
                    rows.append(
                        {
                            "i": int(i),
                            "cx": int(cx),
                            "cy": int(cy),
                            "cz": int(cz),
                            # Sign is not identifiable under squared-radius scoring.
                            # We keep it for the requested hypothesis tuple.
                            "sign": +1,
                            "score": float(scores[iz]),
                            "mean_rt": float(mean_rt[iz]),
                            "std_rt": float(std_rt[iz]),
                        }
                    )

    results_df = pd.DataFrame(rows)

    # Sort exactly as requested
    results_df = results_df.sort_values(
        by=["i", "cx", "cy", "cz", "sign"],
        ascending=True,
        ignore_index=True,
    )

    # Best hypothesis for each i
    best_idx = results_df.groupby("i")["score"].idxmin()
    best_per_i_df = (
        results_df.loc[best_idx]
        .sort_values("i")
        .reset_index(drop=True)
    )

    # Overall winner
    w = results_df.loc[results_df["score"].idxmin()]
    winner = WinningHypothesis(
        i=int(w["i"]),
        cx=int(w["cx"]),
        cy=int(w["cy"]),
        cz=int(w["cz"]),
        sign=int(w["sign"]),
        score=float(w["score"]),
        mean_rt=float(w["mean_rt"]),
        std_rt=float(w["std_rt"]),
    )

    # Console output
    print(f"Loaded: {csv_path}")
    print(f"Samples: {n_samples}")
    print(f"Payload bytes per sample: {n_payload}")
    print(f"Residual mode: {residual_mode}")
    print()
    print("Note: 'sign' is not identifiable under squared-radius scoring.")
    print("      All sign choices are equivalent after squaring, so sign=+1 is used by convention.")
    print()

    print("Top hypotheses by score:")
    print(
        results_df.nsmallest(top_k_print, "score")[
            ["i", "cx", "cy", "cz", "sign", "score", "mean_rt", "std_rt"]
        ].to_string(index=False)
    )
    print()

    print("Lowest-score hypothesis for each i:")
    print(
        best_per_i_df[["i", "cx", "cy", "cz", "sign", "score"]]
        .to_string(index=False)
    )
    print()

    print("Winning hypothesis:")
    print(winner)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        results_df["i"],
        results_df["score"],
        s=5,
        alpha=0.12,
        label="All hypotheses",
    )
    ax.scatter(
        best_per_i_df["i"],
        best_per_i_df["score"],
        s=55,
        marker="o",
        label="Lowest score per i",
    )
    ax.plot(
        best_per_i_df["i"],
        best_per_i_df["score"],
        linewidth=1.2,
        alpha=0.8,
    )
    ax.scatter(
        [winner.i],
        [winner.score],
        s=140,
        marker="*",
        label="Overall winner",
    )

    ax.set_xlabel("Adjacent triple start index i")
    ax.set_ylabel("score = std(rt) / mean(rt)")
    ax.set_title("Accelerometer triple hypothesis search")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_plot_path is not None:
        save_plot_path = Path(save_plot_path)
        fig.savefig(save_plot_path, dpi=160, bbox_inches="tight")
        print(f"\nSaved plot to: {save_plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return results_df, best_per_i_df, winner