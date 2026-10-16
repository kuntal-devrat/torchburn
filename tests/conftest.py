"""Shared pytest configuration for the TorchBurn test suite.

Importing ``torchburn`` here registers the ``torchburn`` Dynamo backend for
*every* test module. Previously each test file had to remember to import
``torchburn`` itself; files that forgot only passed when run after another file
that happened to import it (a hidden ordering dependency that broke isolated
runs like ``pytest tests/test_fallback.py``).
"""

from __future__ import annotations

import torch  # noqa: F401  (imported first: torchburn registers into it)
import torchburn  # noqa: F401  (registers the backend on import)


def pytest_sessionstart(session):  # noqa: D103 (pytest hook)
    # Sanity: the backend must be registered for the whole session.
    assert "torchburn" in torch._dynamo.list_backends()
