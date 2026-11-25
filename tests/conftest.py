import sys
from pathlib import Path

import pytest

# make modules in src/ importable, the same way the scripts see them
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def pytest_addoption(parser):
    parser.addoption("--smoke", action="store_true", help="run the smoke test with real API calls")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--smoke"):
        return
    skip = pytest.mark.skip(reason="smoke test makes paid API calls; run with: pytest --smoke")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip)
