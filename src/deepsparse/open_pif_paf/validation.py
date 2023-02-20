"""Evaluation on COCO data."""
from openpifpaf.eval import cli
from deepsparse.open_pif_paf.utils.validation.deepsparse_evaluator import DeepSparseEvaluator
from deepsparse import Pipeline
def main():
    args = cli()
    args.dataset = "cocokp"
    args.output = "funny"
    args.decoder = ["cifcaf"]
    args.checkpoint = "shufflenetv2k16"
    pipeline = Pipeline.create(task="open_pif_paf", model_path="openpifpaf-resnet50.onnx", output_fields=True)
    evaluator = DeepSparseEvaluator(pipeline = pipeline, dataset_name=args.dataset, skip_epoch0=False, img_size= args.coco_eval_long_edge)
    if args.watch:
        assert args.output is None
        evaluator.watch(args.checkpoint, args.watch)
    else:
        evaluator.evaluate(args.output)


if __name__ == '__main__':
    main()