from collections import defaultdict

_FUNCTIONS_REGISTRY = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: list)))


def register(task, identifier):
    def decorator(f):
        task_registry = _FUNCTIONS_REGISTRY.get(task)
        if task_registry is None:
            _FUNCTIONS_REGISTRY[task][identifier] = [f.__name__]
        else:
            _FUNCTIONS_REGISTRY[task][identifier].append(f.__name__)

        return f

    return decorator