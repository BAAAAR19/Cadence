"""The fitted admission model, on disk.

One file holds everything the gateway needs to reconstruct the bound: the
quantile regressor, the calibration scores it was calibrated on, the level, and
enough provenance to tell which trace and which source tree produced it. The
scores travel with the model rather than being recomputed at start-up because
they are the calibration set -- a bound recalibrated at boot against whatever
data happened to be lying around is a different bound, and it would not be the
one whose coverage the writeup reports.

Pickle, with the sklearn estimator inside it, is the format. That is a real
constraint and it is written down rather than discovered later: the artifact is
only loadable by a compatible scikit-learn, so the version it was fitted with
is recorded in the sidecar JSON and checked, loudly, on load.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1


@dataclass
class PredictorArtifact:
    predictor: Any
    """Anything with ``predict_quantiles(X) -> (lo, hi)``."""
    calib_scores: np.ndarray
    """One-sided nonconformity scores ``y - q_hi(x)`` on the calibration split."""
    alpha: float = 0.01
    score: str = "ratio"
    """Which nonconformity score the calibration scores are in. It travels with
    them because they are meaningless apart: a set of log-ratio scores read as
    absolute seconds would produce a bound of about a second, uniformly, and
    nothing would notice."""
    feature_names: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)
    format_version: int = FORMAT_VERSION

    # --- io ---------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import sklearn

        self.meta.setdefault("saved", datetime.now(UTC).isoformat(timespec="seconds"))
        self.meta["sklearn_version"] = sklearn.__version__
        self.meta["n_calib"] = int(self.calib_scores.size)
        self.meta["alpha"] = self.alpha
        self.meta["score"] = self.score
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "format_version": self.format_version,
                    "alpha": self.alpha,
                    "score": self.score,
                    "feature_names": list(self.feature_names),
                    "calib_score_quantiles": {
                        str(q): float(np.quantile(self.calib_scores, q))
                        for q in (0.5, 0.9, 0.99, 1.0)
                    },
                    **{k: v for k, v in self.meta.items()},
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        return path

    @staticmethod
    def load(path: str | Path) -> PredictorArtifact:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no admission model at {path}. Fit one with "
                f"`uv run bench/fit_predictor.py`, or run with CADENCE_ADMISSION=none."
            )
        with path.open("rb") as fh:
            art = pickle.load(fh)
        if not isinstance(art, PredictorArtifact):
            raise TypeError(f"{path} does not hold a PredictorArtifact")
        if art.format_version != FORMAT_VERSION:
            raise ValueError(
                f"{path} is format v{art.format_version}, this build reads v{FORMAT_VERSION}"
            )
        import sklearn

        want = art.meta.get("sklearn_version")
        if want and want != sklearn.__version__:
            # A warning and not an error: sklearn's pickles usually survive a
            # patch bump, and refusing to serve because of one would be worse
            # than saying so. A silent mismatch would not be.
            print(
                f"warning: admission model was fitted with scikit-learn {want}, "
                f"running {sklearn.__version__}",
                flush=True,
            )
        return art

    def summary(self) -> dict:
        return {"alpha": self.alpha, "n_calib": int(self.calib_scores.size), **self.meta}


__all__ = ["PredictorArtifact", "FORMAT_VERSION"]
