import importlib
import logging
import os
import warnings
import subprocess
import sys
from typing import List, Optional

_LOGGER = logging.getLogger(__name__)

__all__ = ["auto_pip_install", "Dependency", "CheckAttrNotFoundError"]


def auto_pip_install(
    requirer: str,
    *dependencies: "Dependency",
    optional_dependencies: Optional[List["Dependency"]] = None,
):
    """
    Install `dependencies` if they aren't already installed.
    :param requirer: The module that is calling the function - use `__qualname__`.
    :param dependencies: list of required dependencies to install
    :param optional_dependencies: dependencies that are optional to install
    """
    for dependency in dependencies:
        _maybe_pip_install(requirer, dependency, error_on_fail=True)

    if optional_dependencies is not None:
        for dependency in optional_dependencies:
            _maybe_pip_install(requirer, dependency, error_on_fail=False)


def _maybe_pip_install(requirer: str, dependency: "Dependency", *, error_on_fail: bool):
    """
    Install `dependency` using pip. Skips install if environment variable
    `NM_NO_AUTOINSTALL` exists.

    :param requirer: Module that requires the install
    :param dependency: the thing to install
    :param error_on_fail: Whether to raise error or warn if install fails.
    :return: None
    """
    if _get_import_error(dependency) is None:
        _LOGGER.debug(f"{dependency} already installed")
        return

    if os.getenv("NM_NO_AUTOINSTALL", False):
        _LOGGER.warning(
            f"{dependency} not installed."
            "Skipping auto installation due to $NM_NO_AUTOINSTALL."
        )
        return

    if error_on_fail:
        _LOGGER.warning(
            f"{dependency} not installed - auto installing via pip."
            "Set environment variable NM_NO_AUTOINSTALL to disable"
        )

    # get packages to installed & any dependencies of this package
    packages = dependency.packages
    _LOGGER.debug(f"Running `pip install {' '.join(packages)}`")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

    # get the error if we try to import again - if None then it is successful
    import_error = _get_import_error(dependency)
    if import_error is None:
        _LOGGER.info(f"{dependency} (dependency of {requirer}) successfully installed")
        return

    # install failed, warn or raise error depending on necessity of dependency
    msg = (
        f"Unable to import or install {dependency} (a requirement of {requirer})."
        f"Failed with exception: {import_error}"
    )
    if error_on_fail:
        _LOGGER.error(msg)
        raise ValueError(msg)
    else:
        _LOGGER.warning(msg)
        warnings.warn(message=msg, category=UserWarning)


def _get_import_error(dep: "Dependency") -> Optional[ImportError]:
    """
    Try to import `dep` using `importlib`.

    :return: The error if there is any.
    """
    try:
        mod = importlib.import_module(dep.import_name)
        if dep.check_attr is not None and not hasattr(mod, dep.check_attr):
            return CheckAttrNotFoundError(
                f"Unabled to find `{dep.check_attr}` in {dep.import_name}."
                "The wrong version may be installed. Please install using:"
                f"`pip install {' '.join(dep.packages)}`"
            )
        return None
    except ImportError as err:
        return err


class Dependency:
    """
    Represents a dependency that would usually go in a requirements.txt or setup.py
    file.

    :param name: The name to use with `pip install`, can be a url to a .whl
    :param version: Optional appended to `name`. E.g. `==1.2.3` or `>=1.0,<2.0`.
    :param import_name: The name to use when importing
    :param requirements: Optional packages or commands to pass to
    """

    def __init__(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        import_name: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        check_attr: Optional[str] = None,
    ) -> None:
        self.package_name = name
        if version is not None:
            self.package_name += version
        self.import_name = import_name or name
        self.packages = [self.package_name] + (requirements or [])
        self.check_attr = check_attr

    def __str__(self) -> str:
        return self.package_name


class CheckAttrNotFoundError(Exception):
    ...
