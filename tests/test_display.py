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

import time
from unittest.mock import MagicMock

import pytest
from rich.progress import Progress

import langfair.utils.display as display_module
from langfair.utils.display import (
    ConditionalBarColumn,
    ConditionalSpinnerColumn,
    ConditionalTextColumn,
    ConditionalTextPercentageColumn,
    ConditionalTimeElapsedColumn,
)


@pytest.fixture(autouse=True)
def patch_display_progress(monkeypatch):
    monkeypatch.setattr(
        display_module, "start_progress_bar", lambda *args, **kwargs: MagicMock()
    )
    monkeypatch.setattr(
        display_module, "stop_progress_bar", lambda *args, **kwargs: None
    )


# Speed up tests by disabling sleep
@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)


def test_start_progress_bar_without_existing():
    progress = display_module.start_progress_bar()
    assert isinstance(progress, Progress)
    task_id = progress.add_task("[Task]Test", total=10)
    progress.update(task_id, completed=5)
    task = progress.tasks[task_id]
    assert task.completed == 5


def test_start_progress_bar_with_existing():
    existing = Progress()
    progress = display_module.start_progress_bar(existing)
    assert progress is existing
    assert progress.live.is_started


def test_stop_progress_bar_stops():
    progress = display_module.start_progress_bar()
    display_module.stop_progress_bar(progress)
    assert not progress.live.is_started


def test_task_creation_and_update():
    progress = display_module.start_progress_bar()
    task_id = progress.add_task("[Task]Downloading", total=100)
    progress.update(task_id, completed=40)
    task = progress.tasks[task_id]
    assert task.description == "[Task]Downloading"
    assert task.completed == 40
    assert task.total == 100
    display_module.stop_progress_bar(progress)


def test_conditional_columns_render_normal_task():
    progress = display_module.start_progress_bar()
    task_id = progress.add_task("[Task]Processing", total=80)
    progress.update(task_id, completed=20)
    task = progress.tasks[task_id]

    assert ConditionalBarColumn().render(task) != ""
    assert ConditionalTimeElapsedColumn().render(task) != ""
    assert "[progress.description]Processing" in ConditionalTextColumn(
        "[progress.description]{task.description}"
    ).render(task)
    assert "[progress.percentage]20/80" in ConditionalTextPercentageColumn(
        "[progress.percentage]{task.completed}/{task.total}"
    ).render(task)
    assert ConditionalSpinnerColumn().render(task) != ""


def test_conditional_columns_render_no_progress_bar():
    progress = display_module.start_progress_bar()
    task_id = progress.add_task("[No Progress Bar]Hidden", total=50)
    progress.update(task_id, completed=10)
    task = progress.tasks[task_id]

    assert ConditionalBarColumn().render(task) == ""
    assert ConditionalTimeElapsedColumn().render(task) == ""
    assert (
        ConditionalTextColumn("[progress.description]{task.description}").render(task)
        == "[progress.description]Hidden"
    )
    assert (
        ConditionalTextPercentageColumn(
            "[progress.percentage]{task.completed}/{task.total}"
        ).render(task)
        == ""
    )
    assert ConditionalSpinnerColumn().render(task) == ""
