"""Local named-entity recognition for final belief/decision nodes.

The public shape follows Semantica's ``NamedEntityRecognizer`` coordinator, but
this project intentionally supports only non-LLM extraction methods:

``pattern``
    spaCy ``EntityRuler`` patterns (built-in token patterns plus optional custom
    EntityRuler-compatible patterns).
``rules``
    spaCy ``Matcher`` linguistic/token rules.
``ml``
    spaCy statistical NER. This is the default method.
``huggingface``
    A Hugging Face token-classification pipeline.

``regex`` and ``llm`` are deliberately rejected. Entity extraction is invoked by
``stream.py`` only after all merge operations for the current round have
finished and the surviving node text is stable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SUPPORTED_NER_METHODS: tuple[str, ...] = (
    "pattern",
    "rules",
    "ml",
    "huggingface",
)
DEFAULT_NER_METHOD = "ml"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_HUGGINGFACE_MODEL = "dslim/bert-base-NER"


@dataclass(slots=True, frozen=True)
class Entity:
    """One extracted entity mention."""

    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_entity_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a validated, JSON-serialisable NER configuration."""

    raw = dict(config or {})
    method_value = raw.get("method", raw.get("methods", DEFAULT_NER_METHOD))
    if isinstance(method_value, str):
        methods = [method_value]
    elif isinstance(method_value, Sequence):
        methods = [str(value) for value in method_value]
    else:
        methods = [DEFAULT_NER_METHOD]

    methods = [value.strip().lower() for value in methods if str(value).strip()]
    if not methods:
        methods = [DEFAULT_NER_METHOD]
    _validate_methods(methods)

    fallback_value = raw.get("fallback_methods", ["rules", "pattern"])
    if fallback_value is None:
        fallback_methods: list[str] = []
    elif isinstance(fallback_value, str):
        fallback_methods = [fallback_value]
    else:
        fallback_methods = [str(value) for value in fallback_value]
    fallback_methods = [
        value.strip().lower() for value in fallback_methods if str(value).strip()
    ]
    _validate_methods(fallback_methods)

    threshold = float(raw.get("confidence_threshold", raw.get("min_confidence", 0.5)))
    threshold = min(1.0, max(0.0, threshold))

    labels = raw.get("labels")
    if labels is None:
        label_list = None
    elif isinstance(labels, str):
        label_list = [labels.strip().upper()] if labels.strip() else None
    else:
        label_list = [str(value).strip().upper() for value in labels if str(value).strip()]
        label_list = label_list or None

    custom_patterns = raw.get("patterns") or []
    if isinstance(custom_patterns, Mapping):
        custom_patterns = [dict(custom_patterns)]
    elif not isinstance(custom_patterns, list):
        custom_patterns = []

    return {
        "method": methods[0] if len(methods) == 1 else methods,
        "fallback_methods": fallback_methods,
        "confidence_threshold": threshold,
        "merge_overlapping": bool(raw.get("merge_overlapping", True)),
        "include_standard_types": bool(raw.get("include_standard_types", True)),
        "spacy_model": str(raw.get("spacy_model") or raw.get("model") or DEFAULT_SPACY_MODEL),
        "huggingface_model": str(
            raw.get("huggingface_model") or DEFAULT_HUGGINGFACE_MODEL
        ),
        "device": raw.get("device", "auto"),
        "labels": label_list,
        "patterns": custom_patterns,
    }


def _validate_methods(methods: Sequence[str]) -> None:
    unsupported = sorted({method for method in methods if method not in SUPPORTED_NER_METHODS})
    if unsupported:
        supported = ", ".join(SUPPORTED_NER_METHODS)
        raise ValueError(
            "unsupported NER method(s): "
            + ", ".join(unsupported)
            + f". Supported methods are: {supported}. regex and llm are disabled."
        )


class NamedEntityRecognizer:
    """Named-entity recognition coordinator.

    ``method``/``methods`` selects the primary extractor(s). The default is
    ``ml``, implemented with spaCy. If a selected model cannot be loaded, the
    configured non-LLM ``fallback_methods`` are tried in order.
    """

    def __init__(
        self,
        methods: Sequence[str] | None = None,
        confidence_threshold: float = 0.5,
        merge_overlapping: bool = True,
        include_standard_types: bool = True,
        method: Any = None,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        merged: dict[str, Any] = dict(config or {})
        merged.update(kwargs)
        if method is not None:
            merged["method"] = method
        elif methods is not None:
            merged["method"] = list(methods)
        merged.setdefault("confidence_threshold", confidence_threshold)
        merged.setdefault("merge_overlapping", merge_overlapping)
        merged.setdefault("include_standard_types", include_standard_types)

        self.config = normalize_entity_config(merged)
        configured = self.config["method"]
        self.methods = (configured,) if isinstance(configured, str) else tuple(configured)
        self.fallback_methods = tuple(self.config["fallback_methods"])
        self.confidence_threshold = float(self.config["confidence_threshold"])
        self.merge_overlapping = bool(self.config["merge_overlapping"])
        self.include_standard_types = bool(self.config["include_standard_types"])
        self.spacy_model = str(self.config["spacy_model"])
        self.huggingface_model = str(self.config["huggingface_model"])
        self.device = self.config["device"]
        self.allowed_labels = set(self.config["labels"] or []) or None
        self.custom_patterns = list(self.config["patterns"] or [])

        self._spacy_model_nlp = None
        self._spacy_blank_nlp = None
        self._hf_pipeline = None
        self._load_errors: dict[str, str] = {}

    def extract_entities(self, text: str, **options: Any) -> list[Entity]:
        """Extract entities from one English text string."""

        if not isinstance(text, str) or not text.strip():
            return []

        methods = options.get("method", options.get("methods", self.methods))
        if isinstance(methods, str):
            selected = [methods.strip().lower()]
        else:
            selected = [str(value).strip().lower() for value in methods]
        selected = [value for value in selected if value]
        _validate_methods(selected)

        candidates: list[Entity] = []
        attempted: list[str] = []
        primary_succeeded = False

        for extractor_method in selected:
            if extractor_method in attempted:
                continue
            attempted.append(extractor_method)
            try:
                candidates.extend(self._extract_with_method(extractor_method, text))
                primary_succeeded = True
            except Exception as exc:  # optional model/package availability
                self._load_errors[extractor_method] = str(exc)

        # Fallback is for an unavailable/failed primary extractor, not for a
        # valid primary result that simply contains no entities.
        if not primary_succeeded:
            for extractor_method in self.fallback_methods:
                if extractor_method in attempted:
                    continue
                attempted.append(extractor_method)
                try:
                    candidates.extend(self._extract_with_method(extractor_method, text))
                    break
                except Exception as exc:
                    self._load_errors[extractor_method] = str(exc)

        filtered = [
            entity
            for entity in candidates
            if entity.confidence >= self.confidence_threshold
            and (self.allowed_labels is None or entity.label.upper() in self.allowed_labels)
            and self._valid_surface(entity.text)
        ]
        if self.merge_overlapping:
            filtered = self._merge_overlaps(filtered)
        return self._dedupe_in_text_order(filtered)

    def extract_entity_texts(self, text: str, **options: Any) -> list[str]:
        """Return unique entity surface strings in first-appearance order."""

        return [entity.text for entity in self.extract_entities(text, **options)]

    def classify_entities(
        self, entities: Sequence[Entity], **context: Any
    ) -> dict[str, list[Entity]]:
        """Group entities by their normalized label."""

        return EntityClassifier().classify_entities(list(entities), **context)

    def score_confidence(
        self, entities: Sequence[Entity], **options: Any
    ) -> list[Entity]:
        """Return entities with bounded confidence values."""

        return EntityConfidenceScorer().score_entities(list(entities), **options)

    def process_batch(self, texts: Sequence[str], **options: Any) -> list[list[Entity]]:
        return [self.extract_entities(text, **options) for text in texts]

    def extract_entities_batch(
        self, texts: Sequence[str], **options: Any
    ) -> list[list[Entity]]:
        return self.process_batch(texts, **options)

    @property
    def load_errors(self) -> dict[str, str]:
        """Model/package failures encountered during lazy extraction."""

        return dict(self._load_errors)

    def _extract_with_method(self, method: str, text: str) -> Iterable[Entity]:
        if method == "ml":
            return self._extract_ml(text)
        if method == "huggingface":
            return self._extract_huggingface(text)
        if method == "pattern":
            return self._extract_pattern(text)
        if method == "rules":
            return self._extract_rules(text)
        raise ValueError(f"unsupported NER method: {method}")

    def _load_spacy_model(self):
        if self._spacy_model_nlp is not None:
            return self._spacy_model_nlp
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("spaCy is required for the ml NER method") from exc
        try:
            self._spacy_model_nlp = spacy.load(self.spacy_model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {self.spacy_model!r} is not installed. "
                f"Install it with: python -m spacy download {self.spacy_model}"
            ) from exc
        return self._spacy_model_nlp

    def _load_blank_spacy(self):
        if self._spacy_blank_nlp is not None:
            return self._spacy_blank_nlp
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("spaCy is required for pattern/rules NER methods") from exc
        self._spacy_blank_nlp = spacy.blank("en")
        return self._spacy_blank_nlp

    def _extract_ml(self, text: str) -> Iterable[Entity]:
        nlp = self._load_spacy_model()
        doc = nlp(text)
        for span in doc.ents:
            yield Entity(
                text=span.text,
                label=span.label_ or "ENTITY",
                start_char=span.start_char,
                end_char=span.end_char,
                confidence=0.90,
                metadata={"extraction_method": "ml", "model": self.spacy_model},
            )

    def _extract_pattern(self, text: str) -> Iterable[Entity]:
        nlp = self._load_blank_spacy()
        from spacy.pipeline import EntityRuler

        ruler = EntityRuler(nlp, overwrite_ents=True, validate=True)
        patterns: list[dict[str, Any]] = []
        if self.include_standard_types:
            patterns.extend([
                {
                    "label": "ACRONYM",
                    "pattern": [
                        {"IS_UPPER": True, "IS_ALPHA": True},
                        {"IS_UPPER": True, "IS_ALPHA": True, "OP": "*"},
                    ],
                },
                {
                    "label": "PROPER_NAME",
                    "pattern": [
                        {"IS_TITLE": True, "IS_ALPHA": True},
                        {"IS_TITLE": True, "IS_ALPHA": True, "OP": "+"},
                    ],
                },
            ])
        patterns.extend(
            pattern for pattern in self.custom_patterns if isinstance(pattern, dict)
        )
        ruler.add_patterns(patterns)
        doc = nlp(text)
        ruler(doc)
        for span in doc.ents:
            yield Entity(
                text=span.text,
                label=span.label_ or "ENTITY",
                start_char=span.start_char,
                end_char=span.end_char,
                confidence=0.80,
                metadata={"extraction_method": "pattern"},
            )

    def _extract_rules(self, text: str) -> Iterable[Entity]:
        nlp = self._load_blank_spacy()
        from spacy.matcher import Matcher

        doc = nlp(text)
        matcher = Matcher(nlp.vocab)
        if self.include_standard_types:
            matcher.add(
                "PROPER_NAME",
                [[
                    {"IS_TITLE": True, "IS_ALPHA": True},
                    {"IS_TITLE": True, "IS_ALPHA": True, "OP": "*"},
                ]],
            )
            matcher.add(
                "ACRONYM",
                [[{"IS_UPPER": True, "IS_ALPHA": True, "LENGTH": {">=": 2}}]],
            )
            matcher.add("EMAIL", [[{"LIKE_EMAIL": True}]])
            matcher.add("URL", [[{"LIKE_URL": True}]])
        matches = matcher(doc, as_spans=True)
        for span in matches:
            label = span.label_ or "ENTITY"
            yield Entity(
                text=span.text,
                label=label,
                start_char=span.start_char,
                end_char=span.end_char,
                confidence=0.76,
                metadata={"extraction_method": "rules"},
            )

    def _extract_huggingface(self, text: str) -> Iterable[Entity]:
        if self._hf_pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required for the huggingface NER method"
                ) from exc
            device = self._resolve_huggingface_device(self.device)
            self._hf_pipeline = pipeline(
                "token-classification",
                model=self.huggingface_model,
                tokenizer=self.huggingface_model,
                aggregation_strategy="simple",
                device=device,
            )
        for raw in self._hf_pipeline(text):
            start = int(raw.get("start", 0))
            end = int(raw.get("end", start))
            surface = text[start:end] or str(raw.get("word") or "")
            yield Entity(
                text=surface,
                label=str(raw.get("entity_group") or raw.get("entity") or "ENTITY"),
                start_char=start,
                end_char=end,
                confidence=float(raw.get("score", 0.0) or 0.0),
                metadata={
                    "extraction_method": "huggingface",
                    "model": self.huggingface_model,
                },
            )

    @staticmethod
    def _resolve_huggingface_device(value: Any) -> int:
        if isinstance(value, int):
            return value
        text = str(value or "auto").strip().lower()
        if text in {"cpu", "-1"}:
            return -1
        if text.startswith("cuda"):
            if ":" in text:
                try:
                    return int(text.split(":", 1)[1])
                except ValueError:
                    return 0
            return 0
        if text == "auto":
            try:
                import torch

                return 0 if torch.cuda.is_available() else -1
            except ImportError:
                return -1
        try:
            return int(text)
        except ValueError:
            return -1

    @staticmethod
    def _valid_surface(text: str) -> bool:
        clean = str(text or "").strip()
        if not clean or clean.isdigit():
            return False
        return clean.casefold() not in {
            "the",
            "a",
            "an",
            "the user",
            "the assistant",
            "the tool",
            "user",
            "assistant",
            "tool",
        }

    @staticmethod
    def _merge_overlaps(entities: Sequence[Entity]) -> list[Entity]:
        ordered = sorted(
            entities,
            key=lambda entity: (
                entity.start_char,
                -(entity.end_char - entity.start_char),
                -entity.confidence,
            ),
        )
        kept: list[Entity] = []
        for entity in ordered:
            overlapping = [
                other
                for other in kept
                if not (
                    entity.end_char <= other.start_char
                    or entity.start_char >= other.end_char
                )
            ]
            if not overlapping:
                kept.append(entity)
                continue
            best = max(
                [entity, *overlapping],
                key=lambda value: (
                    value.confidence,
                    value.end_char - value.start_char,
                ),
            )
            if best is entity:
                kept = [other for other in kept if other not in overlapping]
                kept.append(entity)
        return sorted(kept, key=lambda value: (value.start_char, value.end_char))

    @staticmethod
    def _dedupe_in_text_order(entities: Sequence[Entity]) -> list[Entity]:
        output: list[Entity] = []
        seen = set()
        for entity in sorted(entities, key=lambda value: (value.start_char, value.end_char)):
            key = (entity.text.casefold(), entity.label.upper())
            if key in seen:
                continue
            seen.add(key)
            output.append(entity)
        return output


class EntityClassifier:
    """Small label-normalization helper mirroring the Semantica coordinator API."""

    _TYPE_HIERARCHY = {
        "PERSON": {"PERSON", "PER"},
        "ORG": {"ORG", "ORGANIZATION"},
        "GPE": {"GPE", "LOCATION", "LOC"},
        "DATE": {"DATE", "TIME"},
        "MONEY": {"MONEY", "CURRENCY"},
        "PERCENT": {"PERCENT", "PERCENTAGE"},
    }

    def classify_entity_type(self, entity: Entity, **context: Any) -> str:
        label = entity.label.upper()
        for canonical, variants in self._TYPE_HIERARCHY.items():
            if label in variants:
                return canonical
        return label

    def classify_entities(
        self, entities: Sequence[Entity], **context: Any
    ) -> dict[str, list[Entity]]:
        grouped: dict[str, list[Entity]] = {}
        for entity in entities:
            grouped.setdefault(self.classify_entity_type(entity, **context), []).append(entity)
        return grouped


class EntityConfidenceScorer:
    """Bound confidence values without changing extraction semantics."""

    def score_entities(
        self, entities: Sequence[Entity], **options: Any
    ) -> list[Entity]:
        output: list[Entity] = []
        for entity in entities:
            confidence = min(1.0, max(0.0, float(entity.confidence)))
            output.append(
                Entity(
                    text=entity.text,
                    label=entity.label,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    confidence=confidence,
                    metadata=dict(entity.metadata),
                )
            )
        return output

