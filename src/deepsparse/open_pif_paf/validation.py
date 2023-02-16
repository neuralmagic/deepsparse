"""Evaluation on COCO data."""
from openpifpaf.eval import cli, Evaluator, Predictor, network, count_ops, __version__, LOG, show
from openpifpaf import  datasets, decoder, visualizer, transforms
from openpifpaf.decoder import CifCaf
import typing as t
import json
from collections import defaultdict
import sys
import os
import time
import PIL.Image
from openpifpaf.decoder.multi import Multi
import argparse

import PIL
import torch

from deepsparse import Pipeline

class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]

def remove_normalization(data_loader: t.Iterable[t.Any]) -> t.Iterable[t.Any]:
    dataset = data_loader.dataset
    if not hasattr(dataset, "preprocess"):
        raise AttributeError("")

    preprocess = dataset.preprocess
    assert len(preprocess.preprocess_list) == 6

    image_preprocess = preprocess.preprocess_list[5]
    assert len(image_preprocess.preprocess_list) == 3

    assert image_preprocess.preprocess_list[2].image_transform.__class__.__name__ == "Normalize"

    preprocess.preprocess_list[5].preprocess_list = image_preprocess.preprocess_list[:2]


    return data_loader

class DeepSparseDecoder(decoder.Decoder):
    def batch(self, pipeline, model, image_batch, *, device=None, gt_anns_batch=None):
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        fields_batch = self.fields_batch(model, image_batch, device=device)
        self.last_nn_time = time.perf_counter() - start_nn

        if gt_anns_batch is None:
            gt_anns_batch = [None for _ in fields_batch]

        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
            image_batch = [None for _ in fields_batch]
            gt_anns_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        start_decoder = time.perf_counter()
        result = self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, image_batch, gt_anns_batch))
        self.last_decoder_time = time.perf_counter() - start_decoder

        LOG.debug('time: nn = %.1fms, dec = %.1fms',
                  self.last_nn_time * 1000.0,
                  self.last_decoder_time * 1000.0)
        return result

def fields_batch_deepsparse_to_torch(fields_batch,device):
    result = []
    fields = fields_batch.fields
    for idx, (cif, caf) in enumerate(zip(*fields)):
        result.append([torch.from_numpy(cif), torch.from_numpy(caf)])
    return result

import numpy
class DeepSparseCifCaf(CifCaf):
    def __init__(self, head_metas):
        cif_metas, caf_metas = head_metas
        super().__init__([cif_metas], [caf_metas])

    def batch(self, pipeline, model, image_batch, *, device=None, gt_anns_batch=None):
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        fields_batch = self.fields_batch(model, image_batch, device=device)
        image_batch = image_batch.numpy() * 255
        image_batch = image_batch[:,:,:480, :640]
        fields_batch_deepsparse= pipeline(images=image_batch.astype(numpy.uint8))
        fields_batch = fields_batch_deepsparse_to_torch(fields_batch_deepsparse, device=device)

        self.last_nn_time = time.perf_counter() - start_nn

        if gt_anns_batch is None:
            gt_anns_batch = [None for _ in fields_batch]

        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
            image_batch = [None for _ in fields_batch]
            gt_anns_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        start_decoder = time.perf_counter()
        result = self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, image_batch, gt_anns_batch))
        self.last_decoder_time = time.perf_counter() - start_decoder

        LOG.debug('time: nn = %.1fms, dec = %.1fms',
                  self.last_nn_time * 1000.0,
                  self.last_decoder_time * 1000.0)
        return result

class DeepSparsePredictor:
    """Convenience class to predict from various inputs with a common configuration."""

    batch_size = 1  #: batch size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  #: device
    fast_rescaling = True  #: fast rescaling
    loader_workers = None  #: loader workers
    long_edge = None  #: long edge

    def __init__(self, checkpoint=None, head_metas=None, *,
                 json_data=False,
                 visualize_image=False,
                 visualize_processed_image=False):
        if checkpoint is not None:
            network.Factory.checkpoint = checkpoint
        self.json_data = json_data
        self.visualize_image = visualize_image
        self.visualize_processed_image = visualize_processed_image

        self.model_cpu, _ = network.Factory().factory(head_metas=head_metas)
        self.model = self.model_cpu.to(self.device)
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
            self.model.base_net = self.model_cpu.base_net
            self.model.head_nets = self.model_cpu.head_nets

        self.preprocess = self._preprocess_factory()
        LOG.debug('head names = %s', [meta.name for meta in head_metas])
        self.processor = DeepSparseCifCaf(self.model_cpu.head_metas)

        self.last_decoder_time = 0.0
        self.last_nn_time = 0.0
        self.total_nn_time = 0.0
        self.total_decoder_time = 0.0
        self.total_images = 0

        LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
                 self.device, torch.cuda.is_available(), torch.cuda.device_count())

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser, *,
            skip_batch_size=False, skip_loader_workers=False):
        """Add command line arguments.

        When using this class together with datasets (e.g. in eval),
        skip the cli arguments for batch size and loader workers as those
        will be provided via the datasets module.
        """
        group = parser.add_argument_group('Predictor')

        if not skip_batch_size:
            group.add_argument('--batch-size', default=cls.batch_size, type=int,
                               help='processing batch size')

        if not skip_loader_workers:
            group.add_argument('--loader-workers', default=cls.loader_workers, type=int,
                               help='number of workers for data loading')

        group.add_argument('--long-edge', default=cls.long_edge, type=int,
                           help='rescale the long side of the image (aspect ratio maintained)')
        group.add_argument('--precise-rescaling', dest='fast_rescaling',
                           default=True, action='store_false',
                           help='use more exact image rescaling (requires scipy)')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Configure from command line parser."""
        cls.batch_size = args.batch_size
        cls.device = args.device
        cls.fast_rescaling = args.fast_rescaling
        cls.loader_workers = args.loader_workers
        cls.long_edge = args.long_edge

    def _preprocess_factory(self):
        rescale_t = None
        if self.long_edge:
            rescale_t = transforms.RescaleAbsolute(self.long_edge, fast=self.fast_rescaling)

        pad_t = None
        if self.batch_size > 1:
            assert self.long_edge, '--long-edge must be provided for batch size > 1'
            pad_t = transforms.CenterPad(self.long_edge)
        else:
            pad_t = transforms.CenterPadTight(16)

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            rescale_t,
            pad_t,
            transforms.EVAL_TRANSFORM,
        ])

    def dataset(self, data):
        """Predict from a dataset."""
        loader_workers = self.loader_workers
        if loader_workers is None:
            loader_workers = self.batch_size if len(data) > 1 else 0

        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.device.type != 'cpu',
            num_workers=loader_workers,
            collate_fn=datasets.collate_images_anns_meta)

        yield from self.dataloader(dataloader)

    def enumerated_dataloader(self, pipeline, enumerated_dataloader):
        """Predict from an enumerated dataloader."""
        for batch_i, item in enumerated_dataloader:
            if len(item) == 3:
                processed_image_batch, gt_anns_batch, meta_batch = item
                image_batch = [None for _ in processed_image_batch]
            elif len(item) == 4:
                image_batch, processed_image_batch, gt_anns_batch, meta_batch = item
            if self.visualize_processed_image:
                visualizer.Base.processed_image(processed_image_batch[0])

            pred_batch = self.processor.batch(pipeline, self.model, processed_image_batch, device=self.device)
            self.last_decoder_time = self.processor.last_decoder_time
            self.last_nn_time = self.processor.last_nn_time
            self.total_decoder_time += self.processor.last_decoder_time
            self.total_nn_time += self.processor.last_nn_time
            self.total_images += len(processed_image_batch)

            # un-batch
            for image, pred, gt_anns, meta in \
                    zip(image_batch, pred_batch, gt_anns_batch, meta_batch):
                LOG.info('batch %d: %s', batch_i, meta.get('file_name', 'no-file-name'))

                # load the original image if necessary
                if self.visualize_image:
                    visualizer.Base.image(image, meta=meta)

                pred = [ann.inverse_transform(meta) for ann in pred]
                gt_anns = [ann.inverse_transform(meta) for ann in gt_anns]

                if self.json_data:
                    pred = [ann.json_data() for ann in pred]

                yield pred, gt_anns, meta

    def dataloader(self, dataloader):
        """Predict from a dataloader."""
        yield from self.enumerated_dataloader(enumerate(dataloader))

    def image(self, file_name):
        """Predict from an image file name."""
        return next(iter(self.images([file_name])))

    def images(self, file_names, **kwargs):
        """Predict from image file names."""
        data = datasets.ImageList(
            file_names, preprocess=self.preprocess, with_raw_image=True)
        yield from self.dataset(data, **kwargs)

    def pil_image(self, image):
        """Predict from a Pillow image."""
        return next(iter(self.pil_images([image])))

    def pil_images(self, pil_images, **kwargs):
        """Predict from Pillow images."""
        data = datasets.PilImageList(
            pil_images, preprocess=self.preprocess, with_raw_image=True)
        yield from self.dataset(data, **kwargs)

    def numpy_image(self, image):
        """Predict from a numpy image."""
        return next(iter(self.numpy_images([image])))

    def numpy_images(self, numpy_images, **kwargs):
        """Predict from numpy images."""
        data = datasets.NumpyImageList(
            numpy_images, preprocess=self.preprocess, with_raw_image=True)
        yield from self.dataset(data, **kwargs)

    def image_file(self, file_pointer):
        """Predict from an opened image file pointer."""
        pil_image = PIL.Image.open(file_pointer).convert('RGB')
        return self.pil_image(pil_image)


class DeepSparseEvaluator(Evaluator):
    def __init__(self, pipeline: Pipeline, dataset_name: str, **kwargs):
        self.pipeline = pipeline
        super().__init__(dataset_name = dataset_name,**kwargs)
        self.data_loader = remove_normalization(self.data_loader)

    def accumulate(self, predictor, metrics):
        prediction_loader = predictor.enumerated_dataloader(self.pipeline, enumerate(self.data_loader))
        if self.loader_warmup:
            LOG.info('Data loader warmup (%.1fs) ...', self.loader_warmup)
            time.sleep(self.loader_warmup)
            LOG.info('Done.')

        total_start = time.perf_counter()
        loop_start = time.perf_counter()

        for image_i, (pred, gt_anns, image_meta) in enumerate(prediction_loader):
            LOG.info('image %d / %d, last loop: %.3fs, images per second=%.1f',
                     image_i, len(self.data_loader), time.perf_counter() - loop_start,
                     image_i / max(1, (time.perf_counter() - total_start)))
            loop_start = time.perf_counter()

            for metric in metrics:
                metric.accumulate(pred, image_meta, ground_truth=gt_anns)

            if self.show_final_image:
                # show ground truth and predictions on original image
                annotation_painter = show.AnnotationPainter()
                with open(image_meta['local_file_path'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

                with show.image_canvas(cpu_image) as ax:
                    if self.show_final_ground_truth:
                        annotation_painter.annotations(ax, gt_anns, color='grey')
                    annotation_painter.annotations(ax, pred)

            if self.n_images is not None and image_i >= self.n_images - 1:
                break
            if image_i == 100:
                break

        total_time = time.perf_counter() - total_start
        return total_time

    def evaluate(self, output: t.Optional[str]):
        # generate a default output filename
        if output is None:
            assert self.args is not None
            output = self.default_output_name(self.args)

        #if self.skip_existing:
        #    stats_file = output + '.stats.json'
        #    if os.path.exists(stats_file):
        #        print('Output file {} exists already. Exiting.'.format(stats_file))
        #        return
        #    print('{} not found. Processing: {}'.format(stats_file, network.Factory.checkpoint))

        predictor = DeepSparsePredictor(head_metas=self.datamodule.head_metas)
        metrics = self.datamodule.metrics()

        total_time = self.accumulate(predictor, metrics)

        # model stats
        counted_ops = list(count_ops(predictor.model_cpu))
        file_size = -1 # TODO

        # write
        additional_data = {
            'args': sys.argv,
            'version': __version__,
            'dataset': self.dataset_name,
            'total_time': total_time,
            'checkpoint': network.Factory.checkpoint,
            'count_ops': counted_ops,
            'file_size': file_size,
            'n_images': predictor.total_images,
            'decoder_time': predictor.total_decoder_time,
            'nn_time': predictor.total_nn_time,
        }

        metric_stats = defaultdict(list)
        for metric in metrics:
            if self.write_predictions:
                metric.write_predictions(output, additional_data=additional_data)

            this_metric_stats = metric.stats()
            assert (len(this_metric_stats.get('text_labels', []))
                    == len(this_metric_stats.get('stats', [])))

            for k, v in this_metric_stats.items():
                metric_stats[k] = metric_stats[k] + v

        stats = dict(**metric_stats, **additional_data)

        # write stats file
        with open(output + '.stats.json', 'w') as f:
            json.dump(stats, f)

        LOG.info('stats:\n%s', json.dumps(stats, indent=4))
        LOG.info(
            'time per image: decoder = %.0fms, nn = %.0fms, total = %.0fms',
            1000 * stats['decoder_time'] / stats['n_images'],
            1000 * stats['nn_time'] / stats['n_images'],
            1000 * stats['total_time'] / stats['n_images'],
        )



def main():
    args = cli()
    args.dataset = "cocokp"
    args.output = "funny"
    args.decoder = ["cifcaf"]
    args.checkpoint = "shufflenetv2k16"
    pipeline = Pipeline.create(task="open_pif_paf", model_path="openpifpaf-resnet50.onnx", output_fields=True)
    #evaluator = Evaluator(dataset_name=args.dataset)
    evaluator = DeepSparseEvaluator(pipeline = pipeline, dataset_name=args.dataset)

    if args.watch:
        assert args.output is None
        evaluator.watch(args.checkpoint, args.watch)
    else:
        evaluator.evaluate(args.output)


if __name__ == '__main__':
    main()