"""Local zero-shot statement-stance classification.

Every semantic source chunk is classified into exactly one of four mutually
exclusive epistemic/utterance stances before its belief, decision, and evidence
records are created:

``asserted``
    A direct, definite statement presented without memory, inference, or
    uncertainty framing.
``recalled``
    A statement explicitly presented as remembered or recalled from past
    experience.
``judged``
    A judgment, assessment, interpretation, conclusion, or inference.
``speculated``
    A guess, possibility, prediction, or otherwise uncertain statement.

The classifier uses an English zero-shot sequence-classification checkpoint.
For every source text it pairs the text with one hypothesis per stance, extracts
the model's entailment logit for each hypothesis, and applies a softmax across
the four stance candidates. This matches the standard single-label zero-shot
classification formulation while keeping all scores available for auditing.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import VALID_STANCES

# Model path is configured per deployment (model_config.yaml -> stance.model_path);
# no machine-specific default is shipped.
DEFAULT_STANCE_MODEL_PATH = ""

STANCE_ORDER = ("asserted", "recalled", "judged", "speculated")

DEFAULT_STANCE_LABELS: dict[str, dict[str, str]] = {
    "asserted": {
        "description": (
            "The speaker presents this as a direct and definite assertion, "
            "without framing it as a memory, inference, judgment, guess, or uncertainty."
        )
    },
    "recalled": {
        "description": (
            "The speaker presents this as something remembered or recalled from "
            "past experience or memory."
        )
    },
    "judged": {
        "description": (
            "The speaker presents this as a judgment, assessment, interpretation, "
            "conclusion, or inference rather than a directly observed fact."
        )
    },
    "speculated": {
        "description": (
            "The speaker presents this as a guess, possibility, prediction, "
            "hypothesis, or otherwise uncertain speculation."
        )
    },
}

_DEFAULT_HYPOTHESIS_TEMPLATE = "{description}"


def normalize_stance_config(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a validated, JSON-serialisable stance classifier config."""
    raw = dict(config or {})
    labels_raw = raw.get("labels")
    if labels_raw is None:
        labels_raw = DEFAULT_STANCE_LABELS
    if not isinstance(labels_raw, Mapping):
        raise ValueError("belief_graph.stance.labels must be an object")

    labels: dict[str, dict[str, str]] = {}
    for stance in STANCE_ORDER:
        default = DEFAULT_STANCE_LABELS[stance]
        value = labels_raw.get(stance, default)
        if isinstance(value, str):
            description = value
        elif isinstance(value, Mapping):
            description = value.get("description") or default["description"]
        else:
            description = default["description"]
        description = str(description or "").strip()
        if not description:
            raise ValueError(
                f"belief_graph.stance.labels.{stance}.description must not be empty"
            )
        labels[stance] = {"description": description}

    unknown = set(labels_raw) - set(STANCE_ORDER)
    if unknown:
        raise ValueError(
            "belief_graph.stance.labels contains unsupported labels: "
            + ", ".join(sorted(str(value) for value in unknown))
        )
    if set(labels) != set(VALID_STANCES):
        raise ValueError(
            "belief_graph.stance.labels must define asserted, recalled, judged, "
            "and speculated exactly once"
        )

    return {
        "enabled": bool(raw.get("enabled", True)),
        "model_path": str(raw.get("model_path") or DEFAULT_STANCE_MODEL_PATH)
        if DEFAULT_STANCE_MODEL_PATH
        else str(raw.get("model_path") or ""),
        "device": str(raw.get("device") or "auto"),
        "dtype": str(raw.get("dtype") or "auto"),
        "batch_size": max(1, int(raw.get("batch_size", 16) or 16)),
        "max_length": max(64, int(raw.get("max_length", 512) or 512)),
        "local_files_only": bool(raw.get("local_files_only", True)),
        "hypothesis_template": str(
            raw.get("hypothesis_template") or _DEFAULT_HYPOTHESIS_TEMPLATE
        ),
        "labels": labels,
    }


@dataclass(frozen=True)
class StancePrediction:
    stance: str
    confidence: float
    scores: dict[str, float]
    model_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stance": self.stance,
            "confidence": self.confidence,
            "scores": dict(self.scores),
            "model_path": self.model_path,
        }


class LocalZeroShotStanceClassifier:
    """Lazy, thread-safe four-class English statement-stance classifier."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = normalize_stance_config(config)
        self.model_path = self.config["model_path"]
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._resolved_device: str | None = None
        self._entailment_index: int | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _resolve_device(self) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        torch = self._torch
        if self.config["device"] == "auto":
            value = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        else:
            value = self.config["device"]
        self._resolved_device = value
        return value

    def _resolve_dtype(self):
        if self._torch is None or self.config["dtype"] in {"", "auto", "none"}:
            return None
        value = getattr(self._torch, self.config["dtype"], None)
        if value is None:
            raise ValueError(
                f"Unsupported stance model dtype: {self.config['dtype']!r}"
            )
        return value

    @staticmethod
    def _canonical_label(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _resolve_entailment_index(self, model_config: Any) -> int:
        id2label = getattr(model_config, "id2label", None) or {}
        if isinstance(id2label, Mapping):
            for raw_idx, raw_label in id2label.items():
                label = self._canonical_label(raw_label)
                if (
                    "entail" in label
                    and not label.startswith(("not_", "non_"))
                    and "not_entail" not in label
                ):
                    try:
                        return int(raw_idx)
                    except (TypeError, ValueError):
                        continue

        label2id = getattr(model_config, "label2id", None) or {}
        if isinstance(label2id, Mapping):
            for raw_label, raw_idx in label2id.items():
                label = self._canonical_label(raw_label)
                if (
                    "entail" in label
                    and not label.startswith(("not_", "non_"))
                    and "not_entail" not in label
                ):
                    try:
                        return int(raw_idx)
                    except (TypeError, ValueError):
                        continue

        num_labels = int(getattr(model_config, "num_labels", 0) or 0)
        if num_labels == 2:
            # Common binary zero-shot checkpoints use [not_entailment, entailment].
            return 1
        if num_labels == 3:
            # Common MNLI checkpoints use [contradiction, neutral, entailment].
            return 2
        raise ValueError(
            "Stance model must expose an entailment label in id2label/label2id, "
            f"or have a supported 2/3-label NLI head; got num_labels={num_labels}."
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Local stance classification requires transformers, torch, and "
                    "sentencepiece. Install them with: "
                    "pip install transformers torch sentencepiece"
                ) from exc

            model_dir = Path(self.model_path)
            if self.config["local_files_only"] and not model_dir.exists():
                raise FileNotFoundError(
                    f"Configured stance model path does not exist: {self.model_path!r}"
                )

            self._torch = torch
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=self.config["local_files_only"],
            )
            model_kwargs: dict[str, Any] = {
                "local_files_only": self.config["local_files_only"],
            }
            dtype = self._resolve_dtype()
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                **model_kwargs,
            )
            entailment_index = self._resolve_entailment_index(model.config)
            model.to(self._resolve_device())
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._entailment_index = entailment_index

    def _hypothesis(self, stance: str) -> str:
        return self.config["hypothesis_template"].format(
            stance=stance,
            label=stance,
            description=self.config["labels"][stance]["description"],
        )

    def classify_texts(self, texts: Sequence[str]) -> list[StancePrediction]:
        """Classify source texts in order and retain all four candidate scores."""
        clean_texts = [str(text or "").strip() for text in texts]
        if not clean_texts:
            return []
        if any(not text for text in clean_texts):
            raise ValueError("Stance classification received an empty source text")
        if not self.config["enabled"]:
            raise RuntimeError(
                "belief_graph.stance.enabled is false, but every generated node requires "
                "a model-inferred stance"
            )

        self._ensure_loaded()
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._entailment_index is not None

        premises: list[str] = []
        hypotheses: list[str] = []
        for text in clean_texts:
            for stance in STANCE_ORDER:
                premises.append(text)
                hypotheses.append(self._hypothesis(stance))

        entailment_logits: list[float] = []
        batch_size = self.config["batch_size"]
        with self._infer_lock:
            for start in range(0, len(premises), batch_size):
                stop = start + batch_size
                encoded = self._tokenizer(
                    premises[start:stop],
                    hypotheses[start:stop],
                    padding=True,
                    truncation="only_first",
                    max_length=self.config["max_length"],
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self._resolve_device())
                    for key, value in encoded.items()
                }
                with torch.inference_mode():
                    logits = self._model(**encoded).logits.float()
                entailment_logits.extend(
                    float(value)
                    for value in logits[:, self._entailment_index]
                    .detach()
                    .cpu()
                    .tolist()
                )

        n_labels = len(STANCE_ORDER)
        predictions: list[StancePrediction] = []
        for text_index in range(len(clean_texts)):
            start = text_index * n_labels
            row = torch.tensor(
                entailment_logits[start : start + n_labels], dtype=torch.float32
            )
            probabilities = torch.softmax(row, dim=-1).tolist()
            scores = {
                stance: round(float(probabilities[index]), 6)
                for index, stance in enumerate(STANCE_ORDER)
            }
            best_index = max(
                range(n_labels),
                key=lambda index: (probabilities[index], -index),
            )
            best_stance = STANCE_ORDER[best_index]
            predictions.append(
                StancePrediction(
                    stance=best_stance,
                    confidence=round(float(probabilities[best_index]), 6),
                    scores=scores,
                    model_path=self.model_path,
                )
            )
        return predictions


_CLASSIFIER_CACHE: dict[str, LocalZeroShotStanceClassifier] = {}
_CLASSIFIER_CACHE_LOCK = threading.Lock()


def get_stance_classifier(
    config: Mapping[str, Any] | None = None,
) -> LocalZeroShotStanceClassifier:
    """Return one shared classifier per normalized config (weights load lazily)."""
    normalized = normalize_stance_config(config)
    cache_key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    with _CLASSIFIER_CACHE_LOCK:
        classifier = _CLASSIFIER_CACHE.get(cache_key)
        if classifier is None:
            classifier = LocalZeroShotStanceClassifier(normalized)
            _CLASSIFIER_CACHE[cache_key] = classifier
        return classifier
