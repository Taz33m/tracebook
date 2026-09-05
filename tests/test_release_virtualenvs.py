"""Release helpers must preserve the platform defaults of `python -m venv`."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import smoke_conformance_wheel, verify_distribution_split


@pytest.mark.parametrize("helper", [smoke_conformance_wheel, verify_distribution_split])
@pytest.mark.parametrize("platform, use_symlinks", [("posix", True), ("nt", False)])
def test_release_helpers_preserve_platform_venv_link_mode(
    tmp_path, monkeypatch, helper, platform, use_symlinks
):
    class EnvironmentRequested(Exception):
        pass

    options = {}
    destinations = []

    class RecordingBuilder:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def create(self, destination):
            destinations.append(Path(destination))
            # Stop before installation or qualification: this unit test checks
            # the actual builder call, not copies of the configuration text.
            raise EnvironmentRequested

    monkeypatch.setattr(helper, "os", SimpleNamespace(name=platform))
    monkeypatch.setattr(helper.venv, "EnvBuilder", RecordingBuilder)

    with pytest.raises(EnvironmentRequested):
        if helper is verify_distribution_split:
            helper._new_environment(tmp_path, "environment")
        else:
            wheel = tmp_path / "example.whl"
            adapter = tmp_path / "adapter.py"
            wheel.write_bytes(b"not installed by this test")
            adapter.write_bytes(b"not executed by this test")
            helper.main([str(wheel), "--adapter", str(adapter)])

    assert options == {"with_pip": True, "clear": True, "symlinks": use_symlinks}
    assert len(destinations) == 1
    assert not destinations[0].exists()
