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
import sqlite3


def con(db_name: str = "cache.sqlite3") -> sqlite3.Connection:
    """Initialize a standardized SQLite3 connection to user's cache directory."""
    cache_dir = platformdirs.user_cache_dir(
        appname="langfair",
        appauthor="cvs-health",
    )

    os.makedirs(cache_dir, exist_ok=True)

    _con = sqlite3.connect(
        database=f"file:{cache_dir}/{db_name}",
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
        uri=True,
    )

    _ = _con.execute("PRAGMA journal_mode=WAL")
    _con.commit()

    return _con
