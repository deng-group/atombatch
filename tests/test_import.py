from importlib.metadata import version

import atombatch


def test_version_matches_package_metadata():
    assert atombatch.__version__ == version("atombatch")
