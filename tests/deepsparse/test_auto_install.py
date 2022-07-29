from deepsparse.auto_install import auto_pip_install, Dependency
from unittest import mock


def test_simple_dependency():
    d = Dependency("my_package")
    assert d.package_name == "my_package"
    assert d.import_name == "my_package"
    assert d.packages == [d.package_name]
    assert d.check_attr is None


def test_versioned_dependency():
    d = Dependency("my_package", version="==1.2.3")
    assert d.package_name == "my_package==1.2.3"
    assert d.import_name == "my_package"
    assert d.packages == [d.package_name]
    assert d.check_attr is None


def test_dependency_import_name():
    d = Dependency("opencv-python", import_name="cv2")
    assert d.package_name == "opencv-python"
    assert d.import_name == "cv2"
    assert d.packages == [d.package_name]
    assert d.check_attr is None


def test_dependency_requirements():
    d = Dependency(
        "my_package", requirements=["other==1.2.3", "-r", "requirements.txt"]
    )
    assert d.package_name == "my_package"
    assert d.import_name == "my_package"
    assert d.packages == [d.package_name, "other==1.2.3", "-r", "requirements.txt"]
    assert d.check_attr is None


@mock.patch("subprocess.check_call")
def test_install_required_raises(install_cmd):
    auto_pip_install("tests", Dependency("fake-package"))
    assert False
