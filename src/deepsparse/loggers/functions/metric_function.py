from .function_registry_base import MetricFunctionRegistryBase
from typing import Any

class MetricFunction(metaclass=MetricFunctionRegistryBase):

    def __call__(self, value: Any):
        """
        Apply the arbitrary function to the value

        :param value: data to apply the function to
        :return: the result
        """
        raise NotImplementedError()



class ChannelMean(MetricFunction):


    def __call__(self, value):
        return value


func = ChannelMean
func(1)