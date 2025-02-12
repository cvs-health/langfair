# Copyright 2024 CVS Health and/or one of its affiliates
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
import platformdirs
import shutil
import sqlite3


CACHE_DIR = platformdirs.user_cache_dir(
    appname="langfair",
    appauthor="cvs-health",
)


_DDL = """
CREATE TABLE IF NOT EXISTS generations (
    generation_id INT PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    generated_text TEXT,
    error TEXT
) STRICT;
"""


def con(db_name: str = "cache.sqlite3") -> sqlite3.Connection:
    """Initialize a standardized SQLite3 connection to user's cache directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    _con = sqlite3.connect(
        database=f"file:{CACHE_DIR}/{db_name}",
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
        uri=True,
    )

    with _con as c:
        _ = c.execute("PRAGMA journal_mode=WAL")
        _ = c.execute(_DDL)

    return _con


def clear() -> None:
    """Completely remove the cache directory."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
