"""Warning filters for noisy third-party compatibility warnings."""

from __future__ import annotations

import logging
import os
import sys
import threading
import warnings

_KNOWN_LOG_MESSAGE_PARTS = (
    "Import deepgemm error for running mxfp4!",
    "Current acext version:",
    "acext token limit:",
    "Disabling overlap schedule since mamba no_buffer is not compatible with "
    "overlap schedule",
    "`torch_dtype` is deprecated! Use `dtype` instead!",
)

_KNOWN_STREAM_MESSAGE_PARTS = (
    "<SAIL>: skipping nsa indexer optimization, fallback to original path",
    "Multi-thread loading shards:",
    "`torch_dtype` is deprecated! Use `dtype` instead!",
)

_KNOWN_NATIVE_MESSAGE_PARTS = _KNOWN_STREAM_MESSAGE_PARTS + (
    "ProcessGroupGloo.cpp:516",
    "Unable to resolve hostname to a (local) address",
    "[Gloo] Rank",
    "Expected number of connected peer ranks is",
)


class _FilteringTextStream:
    def __init__(self, stream, message_parts: tuple[str, ...]):
        self._stream = stream
        self._message_parts = message_parts

    def write(self, text):
        if isinstance(text, str) and any(part in text for part in self._message_parts):
            return len(text)
        return self._stream.write(text)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _install_logging_filter() -> None:
    if getattr(logging.Logger, "_belief_tracer_known_warning_filter", False):
        return

    original_handle = logging.Logger.handle

    def handle(self, record):  # noqa: ANN001
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        if any(part in message for part in _KNOWN_LOG_MESSAGE_PARTS):
            return
        return original_handle(self, record)

    logging.Logger.handle = handle
    logging.Logger._belief_tracer_known_warning_filter = True


def _install_stream_filters() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if getattr(stream, "_belief_tracer_known_warning_filter", False):
            continue
        filtered = _FilteringTextStream(stream, _KNOWN_STREAM_MESSAGE_PARTS)
        filtered._belief_tracer_known_warning_filter = True
        setattr(sys, name, filtered)


def _set_quiet_dependency_env() -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def install_native_output_filter() -> None:
    """Filter known C++/fd-level worker startup noise.

    PyTorch/Gloo and some loaders write directly to file descriptors, bypassing
    Python's warnings, logging, and sys.stderr hooks. This filter is intended
    for SGLang worker processes, not the parent CLI.
    """
    if getattr(install_native_output_filter, "_installed", False):
        return

    try:
        read_fd, write_fd = os.pipe()
        output_fd = os.dup(2)
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
    except OSError:
        return

    native_parts = _KNOWN_NATIVE_MESSAGE_PARTS

    def should_drop(text: str) -> bool:
        return any(part in text for part in native_parts)

    def pump() -> None:
        pending = ""
        while True:
            try:
                chunk = os.read(read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            pending += chunk.decode(errors="replace")
            while True:
                split_at = min(
                    (idx for idx in (pending.find("\n"), pending.find("\r")) if idx >= 0),
                    default=-1,
                )
                if split_at < 0:
                    break
                line = pending[: split_at + 1]
                pending = pending[split_at + 1 :]
                if not should_drop(line) and line.strip():
                    os.write(output_fd, line.encode(errors="replace"))
            if len(pending) > 8192:
                if not should_drop(pending) and pending.strip():
                    os.write(output_fd, pending.encode(errors="replace"))
                pending = ""
        if pending and not should_drop(pending) and pending.strip():
            os.write(output_fd, pending.encode(errors="replace"))

    thread = threading.Thread(
        target=pump,
        name="BeliefTracerNativeOutputFilter",
        daemon=True,
    )
    thread.start()
    install_native_output_filter._installed = True


def suppress_known_warnings() -> None:
    _set_quiet_dependency_env()
    warnings.filterwarnings(
        "ignore",
        message=r'Field "model_(response|output)" has conflict with protected namespace "model_"\.',
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.cuda\.amp\.custom_(fwd|bwd)\(args\.\.\.\)` is deprecated\.",
        category=FutureWarning,
        module=r"megatron\.core\.tensor_parallel\.layers",
    )
    _install_logging_filter()
    _install_stream_filters()


__all__ = ["install_native_output_filter", "suppress_known_warnings"]
