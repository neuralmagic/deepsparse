


class AbstractMiddleware:
    """All middleware should inhertic"""
    ...
    
    def __init__(self):
        self._populate()
        
    def _populate(self):
        # all classes inheriting it class should be mapped
        #
        # some thing like
        #
        # middlewares = [AbstractMiddleware]
        # for klass in middlewares∑
        # self.middlewares = { klass.__name__ : middleware}
        ...