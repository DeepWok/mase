import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runvivado",
        action="store_true",
        default=False,
        help="run tests that require vivado",
    )
    parser.addoption(
        "--runlarge",
        action="store_true",
        default=False,
        help="run tests that require large compute, these are not run by default in our CI",
    )
    parser.addoption(
        "--rundev",
        action="store_true",
        default=False,
        help="run tests that are in dev mode, they may not pass! These are not run by default in our CI",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runvivado"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_vivado = pytest.mark.skip(reason="need --runvivado option to run")
    skip_large = pytest.mark.skip(reason="need --runlarge option to run")
    skip_dev = pytest.mark.skip(reason="need --rundev option to run")
    for item in items:
        if "vivado" in item.keywords:
            item.add_marker(skip_vivado)
        if "large" in item.keywords:
            item.add_marker(skip_large)
        if "dev" in item.keywords:
            item.add_marker(skip_dev)
