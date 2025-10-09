from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def patch_progress_bar(monkeypatch):
    # Original source of progress bar
    import langfair.utils.display as display_module

    monkeypatch.setattr(
        display_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        display_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    # All modules that imported progress bar  directly
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

    monkeypatch.setattr(
        classifier_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        classifier_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        associations_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        associations_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        cooccurrence_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        cooccurrence_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        stereotype_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        stereotype_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        toxicity_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        toxicity_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        cosine_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        cosine_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        sentimentbias_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        sentimentbias_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        bleu_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(bleu_module, "stop_progress_bar", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        rougel_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        rougel_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        counterfactual_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        counterfactual_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        auto_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(auto_module, "stop_progress_bar", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        generator_cf_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        generator_cf_module, "stop_progress_bar", lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        generator_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        generator_module, "stop_progress_bar", lambda *args, **kwargs: None
    )
