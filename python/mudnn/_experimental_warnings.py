"""Warning helpers for experimental cuDNN compatibility APIs."""

import logging

_experimental_api_warnings_emitted = set()


def warn_experimental_api_once(logger: logging.Logger, api_name: str) -> None:
    if api_name in _experimental_api_warnings_emitted:
        return
    _experimental_api_warnings_emitted.add(api_name)
    logger.warning("%s is an experimental API", api_name)


def _reset_experimental_api_warning_registry() -> None:
    _experimental_api_warnings_emitted.clear()
