import os
import importlib

try:
    from deepsparse.cpu import cpu_details
except ImportError:
    raise ImportError(
        "Unable to import deepsparse python apis. "
        "Please contact support@neuralmagic.com"
    )

CORES_PER_SOCKET, AVX_TYPE, VNNI = cpu_details()


def import_deepsparse_engine():
    try:
        nm_package_dir = os.path.dirname(os.path.abspath(__file__))
        onnxruntime_neuralmagic_so_path = os.path.join(
            nm_package_dir, AVX_TYPE, "deepsparse_engine.so"
        )
        spec = importlib.util.spec_from_file_location(
            "deepsparse.{}.deepsparse_engine".format(AVX_TYPE),
            onnxruntime_neuralmagic_so_path,
        )
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        return engine
    except ImportError:
        raise ImportError(
            "Unable to import deepsparse engine binaries. "
            "Please contact support@neuralmagic.com"
        )
