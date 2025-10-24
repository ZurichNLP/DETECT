import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Your Lightning model
from lens.lens.models.detect_metric import DetectMetric


def _gaussian_cdf_rescale(z: float) -> float:
    """Rescale z to [0, 100] via Gaussian CDF."""
    return 0.5 * (math.erf(z / math.sqrt(2)) + 1.0) * 100.0


class DETECT:
    """
    Lightweight user-facing wrapper for DETECT (LENS family).

    Usage
    -----
    detect = DETECT(rescale=True)  # downloads HF ckpt
    scores = detect.score(srcs, mts, refs, batch_size=8, devices=[0])

    Returns
    -------
    List[Dict[str, float]]
        For each item, a dict with:
            - 'simplicity'
            - 'meaning_preservation'
            - 'fluency'
            - 'total'  (rule: if any < 25 -> min; else 0.4*S + 0.4*M + 0.2*F)
    """

    def __init__(
        self,
        *,
        rescale: bool = True,
        map_location: Optional[str] = None,
        strict: bool = True,
        repo_id: str = "ZurichNLP/DETECT",
        filename: str = "best-LENS_multi_wechsel_reducedhs-epoch=04.ckpt",
    ) -> None:
        self.rescale = rescale
        self._device: Optional[torch.device] = None

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Load the Lightning module
        initial_map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: DetectMetric = DetectMetric.load_from_checkpoint(
            ckpt_path, strict=strict, map_location=initial_map_location
        )
        self.model.eval()

    @staticmethod
    def total(s: float, m: float, f: float) -> float:
        """
        LENS total:
          - if min(s,m,f) < 25 -> total = min(s,m,f)
          - else total = 0.4*s + 0.4*m + 0.2*f
        """
        mn = min(s, m, f)
        if mn < 25:
            return float(mn)
        return float(0.4 * s + 0.4 * m + 0.2 * f)

    @torch.inference_mode()
    def score(
        self,
        complex: Sequence[str],
        simple: Sequence[str],
        references: Sequence[Sequence[str]],
        *,
        features: Optional[Sequence[Optional[np.ndarray]]] = None,  # optional per item
        batch_size: int = 8,
        devices: Optional[Sequence[int]] = None,
        show_progress: bool = False,
    ) -> List[Dict[str, float]]:
        """
        Compute DETECT scores.

        Parameters
        ----------
        complex : list[str]
            Source (original/complex) texts.
        simple : list[str]
            System outputs (simplified).
        references : list[list[str]]
            `references[i]` is a list of reference simplifications for item i.
        features : Optional[list[np.ndarray or None]]
            Optional extra features per item (if your model expects them).
        batch_size : int
            Iteration chunk size (we still do per-sample ref expansion internally).
        devices : Optional[list[int]]
            CUDA device IDs; the first available will be used. Falls back to CPU.
        show_progress : bool
            If True, show a tqdm progress bar (if installed).

        Returns
        -------
        list of dict with keys: 'simplicity', 'meaning_preservation', 'fluency', 'total'
        """
        self._set_device_from_devices(devices)
        n = self._validate_lengths(complex, simple, references)

        if features is not None and len(features) != n:
            raise ValueError(
                f"'features' length must match inputs: got {len(features)} vs {n}"
            )

        iterator: Iterable[int] = range(0, n, max(1, batch_size))
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore
                iterator = tqdm(iterator, desc="Scoring")
            except Exception:
                pass

        results: List[Dict[str, float]] = []
        self.model.to(self._device or "cpu")

        for start in iterator:
            end = min(start + batch_size, n)
            for i in range(start, end):
                sample = self._make_sample(
                    complex[i],
                    simple[i],
                    references[i],
                    feats=(None if features is None else features[i]),
                )

                # Expand into per-reference inputs via Lightning hook
                prepared = self.model.prepare_sample([sample], stage="predict")
                # We score one item at a time, but prepared may contain 1 dict that already
                # includes the ref dimension; some implementations produce multiple dicts.
                # We support both: if multiple dicts, run once and stack; if single dict, just run.
                if isinstance(prepared, (list, tuple)) and len(prepared) > 1:
                    # (Uncommon) multiple ref-wise batches
                    all_scores = []
                    for ref_input in prepared:
                        batched = {k: v.to(self._device or "cpu") for k, v in ref_input.items()}
                        out = self.model.forward(**batched)
                        # expected shape: [n_refs, 3] or [1, 3]
                        s = out["scores"].detach().cpu().numpy().astype("float64")
                        all_scores.append(s)
                    scores_per_ref = np.concatenate(all_scores, axis=0)
                else:
                    ref_input = prepared[0] if isinstance(prepared, (list, tuple)) else prepared
                    batched = {k: v.to(self._device or "cpu") for k, v in ref_input.items()}
                    out = self.model.forward(**batched)
                    scores_per_ref = out["scores"].detach().cpu().numpy().astype("float64")  # [n_refs, 3]

                if self.rescale:
                    scores_per_ref = np.vectorize(_gaussian_cdf_rescale)(scores_per_ref)

                # Average across references -> [3]
                mean3 = scores_per_ref.mean(axis=0)
                s_val = float(mean3[0])  # simplicity
                m_val = float(mean3[1])  # meaning preservation
                f_val = float(mean3[2])  # fluency
                total_val = self.total(s_val, m_val, f_val)

                results.append(
                    {
                        "simplicity": s_val,
                        "meaning_preservation": m_val,
                        "fluency": f_val,
                        "total": total_val,
                    }
                )

        return results

    # ------------------ helpers ------------------

    def _set_device_from_devices(self, devices: Optional[Sequence[int]]) -> None:
        """Pick the first usable CUDA device from `devices`, else CPU."""
        if devices and torch.cuda.is_available():
            for d in devices:
                if isinstance(d, int) and 0 <= d < torch.cuda.device_count():
                    self._device = torch.device(f"cuda:{d}")
                    break
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _validate_lengths(
        complex: Sequence[str],
        simple: Sequence[str],
        references: Sequence[Sequence[str]],
    ) -> int:
        if not (len(complex) == len(simple) == len(references)):
            raise ValueError(
                "Input length mismatch: 'complex', 'simple', and 'references' "
                f"must have the same length. Got {len(complex)}, {len(simple)}, {len(references)}."
            )
        for idx, refs in enumerate(references):
            if not isinstance(refs, (list, tuple)) or len(refs) == 0:
                raise ValueError(f"'references[{idx}]' must be a non-empty list of strings.")
            if not all(isinstance(r, str) for r in refs):
                raise TypeError(f"All elements of 'references[{idx}]' must be strings.")
        return len(complex)

    @staticmethod
    def _make_sample(
        src: str,
        mt: str,
        refs: Sequence[str],
        *,
        feats: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if not isinstance(src, str):
            raise TypeError("'complex' entries must be strings.")
        if not isinstance(mt, str):
            raise TypeError("'simple' entries must be strings.")
        if not refs:
            raise ValueError("Each item must have at least one reference string.")

        sample: Dict[str, Any] = {"src": src, "mt": mt, "ref": list(refs)}
        if feats is not None:
            sample["features"] = feats
        return sample