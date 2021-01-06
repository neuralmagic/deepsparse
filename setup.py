from typing import Tuple, List, Dict
from setuptools import find_packages, setup
from setuptools.command.install import install
from fnmatch import fnmatch
from distutils import log
import os

# File regexes for binaries to include in package_data
binary_regexes = ["*/*.so", "*/*.so.*", "*.bin", "*/*.bin"]

class OverrideInstall(install):
    """
    This class adds a hook that runs after regular install that
    changes the permissions of all the binary files to 0755.
    """
    def run(self):
        install.run(self)
        mode = 0o755
        for filepath in self.get_outputs():
            if any(fnmatch(filepath, regex) for regex in binary_regexes):
                log.info("changing mode of %s to %s" % (filepath, oct(mode)))
                os.chmod(filepath, mode)


def _setup_package_dir() -> Dict:
    return {"": "src"}


def _setup_packages() -> List:
    return find_packages(
        "src", include=["nmie", "nmie.*"], exclude=["*.__pycache__.*"]
    )


def _setup_package_data() -> Dict:
    return {"nmie": binary_regexes}


def _setup_install_requires() -> List:
    return [
        "numpy>=1.16.3",
        "onnx>=1.5.0,<1.8.0",
        "requests>=2.0.0"
    ]


def _setup_extras() -> Dict:
    return {}


def _setup_entry_points() -> Dict:
    return {}


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"


setup(
    name="nmie",
    version="0.1.0",
    author="Bill Nell, Michael Goin, Mark Kurtz",
    author_email="support@neuralmagic.com",
    description="The high performance Neural Magic Inference Engine designed "
    "for running deep learning on X86 CPU architectures",
    long_description=_setup_long_description()[0],
    long_description_content_type=_setup_long_description()[1],
    keywords="inference machine learning x86 x86_64 avx2 avx512 neural network",
    license="[TODO]",
    url="https://github.com/neuralmagic/engine",
    package_dir=_setup_package_dir(),
    include_package_data=True,
    package_data=_setup_package_data(),
    packages=_setup_packages(),
    install_requires=_setup_install_requires(),
    extras_require=_setup_extras(),
    entry_points=_setup_entry_points(),
    python_requires=">=3.6.0",
    classifiers=[
        "[TODO]"
    ],
    cmdclass={"install": OverrideInstall},
)
