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


from rich.progress import BarColumn, SpinnerColumn, TextColumn, TimeElapsedColumn


class ConditionalBarColumn(BarColumn):
    def render(self, task):
        if task.description.startswith("[No Progress Bar]"):
            return ""
        return super().render(task)


class ConditionalTimeElapsedColumn(TimeElapsedColumn):
    def render(self, task):
        if task.description.startswith("[No Progress Bar]"):
            return ""
        return super().render(task)


class ConditionalTextColumn(TextColumn):
    def render(self, task):
        if task.description.startswith(
            "[No Progress Bar]"
        ) or task.description.startswith("[Task]"):
            return f"[progress.description]{task.description.replace('[No Progress Bar]', '').replace('[Task]', '')}"
        return super().render(task)


class ConditionalTextPercentageColumn(TextColumn):
    def render(self, task):
        if task.description.startswith("[No Progress Bar]"):
            return ""
        elif task.description.startswith("[Task]"):
            return f"[progress.percentage]{task.completed}/{task.total}"
        return super().render(task)


class ConditionalSpinnerColumn(SpinnerColumn):
    def render(self, task):
        if task.description.startswith("[No Progress Bar]"):
            return ""
        return super().render(task)
