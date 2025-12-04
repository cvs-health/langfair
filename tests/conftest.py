# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest
from unittest.mock import MagicMock


def _should_skip_monkeypatch(request) -> bool:
    """Return True if we should NOT monkeypatch progress bars for this test."""

    # 1) Explicit marker on test or file: @pytest.mark.real_progress
    if request.node.get_closest_marker("real_progress"):
        return True

    # 2) Environment variable override (global opt-out)
    if os.environ.get("PYTEST_REAL_PROGRESS") == "1":
        return True

    # 3) filename/module checks
    fspath = getattr(request.node, "fspath", None)
    if fspath is not None and "test_display" in str(fspath):
        return True

    module = getattr(request.node, "module", None)
    module_name = getattr(module, "__name__", "")
    if module_name.endswith("test_display"):
        return True

    return False


@pytest.fixture(autouse=True)
def patch_progress_bar(monkeypatch, request):
    """
    Monkeypatch start_progress_bar and stop_progress_bar for most tests
    to avoid rendering overhead. Skip monkeypatching when:
      - the test (or file) is marked with @pytest.mark.real_progress
      - PYTEST_REAL_PROGRESS=1 in the environment
      - the file/module name matches `test_display` (fallback)
    """
    if _should_skip_monkeypatch(request):
        return

    # Apply monkeypatch for display module
    import langfair.utils.display as display_module

    monkeypatch.setattr(
        display_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        display_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    # Apply monkeypatch for all modules that imported progress bar directly
    import langfair.auto.auto as auto_module
    import langfair.generator.counterfactual as generator_cf_module
    import langfair.generator.generator as generator_module
    import langfair.metrics.counterfactual.counterfactual as counterfactual_module
    import langfair.metrics.counterfactual.metrics.bleu as bleu_module
    import langfair.metrics.counterfactual.metrics.cosine as cosine_module
    import langfair.metrics.counterfactual.metrics.rougel as rougel_module
    import langfair.metrics.counterfactual.metrics.sentimentbias as sentimentbias_module
    import langfair.metrics.stereotype.metrics.associations as associations_module
    import langfair.metrics.stereotype.metrics.classifier as classifier_module
    import langfair.metrics.stereotype.metrics.cooccurrence as cooccurrence_module
    import langfair.metrics.stereotype.stereotype as stereotype_module
    import langfair.metrics.toxicity.toxicity as toxicity_module

    for module in (
        auto_module,
        generator_cf_module,
        generator_module,
        counterfactual_module,
        bleu_module,
        cosine_module,
        rougel_module,
        sentimentbias_module,
        associations_module,
        classifier_module,
        cooccurrence_module,
        stereotype_module,
        toxicity_module,
    ):
        monkeypatch.setattr(
            module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
        )
        monkeypatch.setattr(module, "stop_progress_bar", lambda *args, **kwargs: None)


def pytest_configure(config):
    # Register the marker so pytest doesn't warn
    config.addinivalue_line(
        "markers",
        "real_progress: Run test without monkeypatching progress bars (use real rich.Progress).",
    )
