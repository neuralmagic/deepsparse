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
    """
    Compares two lists of output tensors, checking that they are sufficiently similar
    :param outputs: List of numpy arrays, usually model outputs
    :param gt_outputs: List of numpy arrays, usually reference outputs
    :param atol: Absolute tolerance for allclose
    :param rtol: Relative tolerance for allclose
    :return: The list of max differences for each pair of outputs
    """
    max_diffs = []

    if len(outputs) != len(gt_outputs):
        raise Exception(
            f"ERROR: number of outputs doesn't match, {len(outputs)} != {len(gt_outputs)}"
        )

    for i in range(len(gt_outputs)):
        gt_output = gt_outputs[i]
        output = outputs[i]

        if output.shape != gt_output.shape:
            raise Exception(
                f"ERROR: output shapes don't match, {output.shape} != {gt_output.shape}"
            )
        if type(output) != type(gt_output):
            raise Exception(
                f"ERROR: output types don't match, {type(output)} != {type(gt_output)}"
            )

        max_diff = numpy.max(numpy.abs(output - gt_output))
        max_diffs.append(max_diff)
        log.info(f"output {i}: {output.shape} {gt_output.shape} MAX DIFF: {max_diff}")
        if not numpy.allclose(output, gt_output, rtol=rtol, atol=atol):
            raise Exception(
                f"ERROR: output data doesn't match\n"
                f"output {i}: {output.shape} {gt_output.shape} MAX DIFF: {max_diff}"
            )

    return max_diffs
