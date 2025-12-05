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

from types import SimpleNamespace

import pytest


class FakeTask:
    def __init__(self, task_id, description, total):
        self.id = task_id
        self.description = description
        self.total = total
        self.completed = 0


class FakeProgress:
    """
    Minimal stand-in for rich.progress.Progress used in tests.
    - add_task(description, total) -> task_id
    - update(task_id, completed=...)
    - tasks[task_id] -> FakeTask
    - live.is_started -> bool
    """

    def __init__(self):
        self._next_id = 0
        self.tasks = {}
        self.live = SimpleNamespace(is_started=False)

    def add_task(self, description, total):
        task_id = self._next_id
        self._next_id += 1
        self.tasks[task_id] = FakeTask(task_id, description, total)
        return task_id

    def update(self, task_id, completed=None):
        task = self.tasks[task_id]
        if completed is not None:
            task.completed = completed

    def start(self):
        self.live.is_started = True

    def stop(self):
        self.live.is_started = False


@pytest.fixture(autouse=True)
def mock_display_progress(monkeypatch):
    """
    Mock progress helpers globally so tests never touch Rich's Live display.
    """
    import langfair.utils.display as display_module

    def _start_progress_bar(existing_progress_bar=None):
        if isinstance(existing_progress_bar, FakeProgress):
            existing_progress_bar.start()
            return existing_progress_bar
        fake = FakeProgress()
        fake.start()
        return fake

    def _stop_progress_bar(progress_bar):
        if isinstance(progress_bar, FakeProgress):
            progress_bar.stop()

    monkeypatch.setattr(display_module, "start_progress_bar", _start_progress_bar)
    monkeypatch.setattr(display_module, "stop_progress_bar", _stop_progress_bar)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_progress: Opt-out of the FakeProgress mock (use real rich.Progress).",
    )
