from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def patch_progress_bar(monkeypatch):
    import langfair.utils.display as display_module

    monkeypatch.setattr(
        display_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        display_module, "stop_progress_bar", lambda *args, **kwargs: None
    )
