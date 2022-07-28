import importlib
import logging
import os
import warnings
import subprocess
import sys
from typing import List, Optional

_LOGGER = logging.getLogger(__name__)

__all__ = ["auto_pip_install", "Dependency", "CheckAttrNotFoundError"]


def auto_pip_install(requirer: str, *dependencies: "Dependency"):
    for dependency in dependencies:
        _maybe_pip_install(requirer, dependency)


def _maybe_pip_install(requirer: str, dependency: "Dependency"):
    if _get_import_error(dependency) is None:
        _LOGGER.debug(f"{dependency} already installed")
        return

    # skip if user has NM_NO_AUTOINSTALL set - warn either way
    if os.getenv("NM_NO_AUTOINSTALL", False):
        _LOGGER.warning(
            f"{dependency} not installed."
            "Skipping auto installation due to $NM_NO_AUTOINSTALL."
        )
        return

    if dependency.necessary:
        _LOGGER.warning(
            f"{dependency} not installed - auto installing via pip."
            "Set environment variable NM_NO_AUTOINSTALL to disable"
        )

    # get packages to installed & any dependencies of this package
    packages = " ".join([dependency.package_name] + dependency.requirements)

    _LOGGER.debug(f"Running `pip install {packages}`")
    subprocess.check_call([sys.executable, "-m", "pip", "install", packages])

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
    if dependency.necessary:
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
        mod = importlib.import_module(dep.import_name or dep.name)
        if dep.check_attr is not None and not hasattr(mod, dep.check_attr):
            return CheckAttrNotFoundError(
                f"Unabled to find `{dep.check_attr}` in {dep.import_name}."
                "The wrong version may be installed. Please install using:"
                f"`pip install {dep.package_name}`"
            )
        return None
    except ImportError as err:
        return err


class Dependency:
    def __init__(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        necessary: bool = True,
        import_name: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        check_attr: Optional[str] = None,
    ) -> None:
        self.name = name
        self.version = version
        self.necessary = necessary
        self.error_on_bad_install = necessary
        self.import_name = import_name
        self.check_attr = check_attr
        self.requirements = requirements

    @property
    def package_name(self) -> str:
        if self.version is None:
            return self.name
        return f"{self.name}{self.version}"

    def __str__(self) -> str:
        return self.package_name


class CheckAttrNotFoundError(Exception):
    ...
