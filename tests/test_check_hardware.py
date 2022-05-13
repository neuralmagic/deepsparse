import pytest

from helpers import run_command


@pytest.mark.cli
def test_check_hardware():
    cmd = ["deepsparse.check_hardware"]
    print(f"\n==== deepsparse.check_hardware command ====\n{' '.join(cmd)}")

    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== deepsparse.check_hardware output ====\n{res.stdout}")

    assert res.returncode == 0, "command exited with non-zero status"
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()

    # check for (static portions of) expected lines
    assert "DeepSparse FP32 model performance supported:" in res.stdout
    assert "DeepSparse INT8 (quantized) model performance supported:" in res.stdout
    assert "Additional CPU info:" in res.stdout
