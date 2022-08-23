from deepsparse.timing.timing_schema import InferenceTimingSchema
from typing import List

def consolidate_batch_timing(batched_inference_timing: List[InferenceTimingSchema], consolidation_func = max):
    test = batched_inference_timing * 2
    timings = [dict(batch_timing) for batch_timing in test]
    aggregated_metrics = {}
    for metric_name in timings[0].keys():
        aggregated_metrics[metric_name] = consolidation_func(batch_timing[metric_name] for batch_timing in timings)

    return InferenceTimingSchema(**aggregated_metrics)

