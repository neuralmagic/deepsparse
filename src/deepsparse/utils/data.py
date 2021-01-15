import os
from typing import List

import numpy

from deepsparse.utils.log import log_init


__all__ = ["verify_outputs"]

log = log_init(os.path.basename(__file__))


def verify_outputs(
    outputs: List[numpy.array],
    gt_outputs: List[numpy.array],
    atol: float = 8.0e-4,
    rtol: float = 0.0,
) -> List[float]:
    max_diffs = []
    assert len(gt_outputs) == len(outputs)
    for i in range(len(gt_outputs)):
        gt_output = gt_outputs[i]
        output = outputs[i]

        assert gt_output.shape == output.shape
        assert type(gt_output) == type(output)

        max_diff = numpy.max(numpy.abs(output - gt_output))
        max_diffs.append(max_diff)
        print(
            "    output {}: {} {} MAX DIFF: {}".format(
                i, gt_output.shape, output.shape, max_diff
            )
        )
        if not numpy.allclose(output, gt_output, rtol=rtol, atol=atol):
            raise Exception(
                "ERROR: output {}: {} {} MAX DIFF: {}".format(
                    i, gt_output.shape, output.shape, max_diff
                )
            )

    return max_diffs
