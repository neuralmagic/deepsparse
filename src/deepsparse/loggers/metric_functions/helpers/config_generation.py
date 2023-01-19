__all__ = ["data_logging_config_from_predefined"]

from typing import Dict, List, Optional, Union, Any

from deepsparse.loggers.build_logger import default_logger


def data_logging_config_from_predefined(
    group_names: Union[str, List[str]],
    loggers: Dict[str, Optional[Dict[str, Any]]] = default_logger(),
    frequency: int = 1,
    save_dir: Optional[str] = None,
    save_name: str = "data_logging_config.yaml",
) -> str:
    """
    Generate a data logging config yaml string using a predefined
    function groups configuration.

    :param group_names: A single group name or a list of group names,
        that are to be translated into the yaml configuration.
    :param loggers: Defines set of loggers that will be used to collect
        the data logs. It is dictionary that maps the logger integration
        names to their initialization arguments
    :param frequency: Optional frequency of the data logging
        functions in the resulting yaml configuration. By default,
        set to 1
    :param save_dir: If provided, the resulting yaml configuration is
        saved to the provided directory
    :param save_name: If config is saved, it will be under this
        filename
    :return: A string yaml dict that specifies the data logging
        configuration
    """
    raise NotImplementedError()