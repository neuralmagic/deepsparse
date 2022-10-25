from typing import Any

class MetricFunctionRegistryBase(type):

    METRIC_FUNCTION_REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.METRIC_FUNCTION_REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    def get_metric_function_registry(cls):
        return dict(cls.METRIC_FUNCTION_REGISTRY)


class MetricFunction(metaclass=MetricFunctionRegistryBase):

    def __call__(self, value: Any):
        """
        Apply the arbitrary function to the value

        :param value: data to apply the function to
        :return: the result
        """
        raise NotImplementedError()



class Channel_Mean(MetricFunction):


    def __call__(self, value):
        return value


func = Channel_Mean()
func(1)