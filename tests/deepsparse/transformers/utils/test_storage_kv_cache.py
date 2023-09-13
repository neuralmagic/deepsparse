# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from deepsparse.transformers.utils.storage_kv_cache import SessionStorageKVCache


class DummyDecoderKVCache:
    def __init__(self, session_id):
        self.id = session_id
        self.engine_internal_cache = None


class TestSessionStorageKVCache:
    storage = SessionStorageKVCache()

    storage.put(DummyDecoderKVCache(session_id="first_session"))
    storage.put(DummyDecoderKVCache(session_id="second_session"))
    assert len(storage) == 2

    # overwrite a session
    storage.put(DummyDecoderKVCache(session_id="first_session"))
    assert len(storage) == 2

    def test_str(self):
        assert str(self.storage)

    def test_has_session(self):
        assert self.storage.has_session("first_session")
        assert self.storage.has_session("second_session")
        assert not self.storage.has_session("third_session")

    def test_get(self):
        assert self.storage.get("first_session")
        assert self.storage.get("second_session")
        assert not self.storage.get("third_session")

    def test_internal_cache_active(self):
        assert not self.storage.internal_cache_active

    def test_pop(self):
        self.storage.put(DummyDecoderKVCache(session_id="session_to_pop"))
        assert self.storage.has_session("session_to_pop")
        self.storage.pop("session_to_pop")
        assert not self.storage.has_session("session_to_pop")
