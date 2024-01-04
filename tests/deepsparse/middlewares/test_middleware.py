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


import copy

from deepsparse.middlewares import MiddlewareManager, MiddlewareSpec
from tests.deepsparse.middlewares.utils import DummyMiddleware, ReducerMiddleware


def test_middleware_spec():
    """Check the __iter__ in MiddlewareSpec"""
    kwargs = dict(foo="bar")
    middleware = [MiddlewareSpec(DummyMiddleware, **kwargs)]

    for mw, init_args in middleware:
        assert mw == DummyMiddleware
        assert init_args == kwargs


def test_middleware_manager_initialization():
    """check that middleware's .send() is defined in init"""
    middleware = [MiddlewareSpec(DummyMiddleware)]
    dummy_middleware_send = DummyMiddleware.send

    # DummyMiddleware.send should get overwritten
    middleware_manager = MiddlewareManager(middleware)

    assert middleware_manager.middleware[0].cls.send is not dummy_middleware_send


def test_middleware_manager_add_middleware():
    """Check the functionality of add_middleware"""
    middleware = []
    middleware_manager = MiddlewareManager(middleware)
    dummy_middleware_send = DummyMiddleware.send

    assert len(middleware_manager.middleware) == 0

    middleware_manager.add_middleware([MiddlewareSpec(DummyMiddleware)])
    assert len(middleware_manager.middleware) == 1

    assert middleware_manager.middleware[-1].cls.send is not dummy_middleware_send


def test_middleware_manager_recieve():
    """
    Check that the middleware's .send() is overwritten and
    writes its update to the middleware state
    """
    reducer_middleware = ReducerMiddleware
    middleware = [MiddlewareSpec(ReducerMiddleware)]
    middleware_manager = MiddlewareManager(middleware)

    original_state = copy.deepcopy(middleware_manager.state)

    # intialize the middleware
    initalized_reducer_middleware = reducer_middleware()
    # trigger the callable
    initalized_reducer_middleware()
    assert middleware_manager.state != original_state
    assert "ReducerMiddleware" in middleware_manager.state
    assert middleware_manager.state["ReducerMiddleware"] == [1]
