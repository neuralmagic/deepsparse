import glob
from typing import Tuple, Iterator, Optional, Union, Iterable, Any, List
import time
from pathlib import Path
import logging
import os
import numpy
import shutil
from sparsezoo.utils import create_dirs

try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error

_LOGGER = logging.getLogger(__name__)

def get_image_loader_and_saver(
    path: str,
    save_dir: str,
    image_shape: Tuple[int, int] = (640, 640),
    target_fps: Optional[float] = None,
    no_save: bool = False,
    batch_size: int = 1,
) -> Union[Iterable, Any, bool]:
    """

    :param path: file path to image or directory of .jpg files, a .mp4 video,
        or an integer (i.e. 0) for web-cam
    :param save_dir: path of directory to save to
    :param image_shape: size of input image_batch to model
    :param target_fps: fps to save potential video at
    :param no_save: set true if not saving results of processing
    :return: image loader iterable, result saver objects
        image_batch, video, or web-cam based on path given, and a boolean value
        that is True is the returned objects load videos
    """
    # video
    if path.endswith(".mp4"):
        loader = VideoLoader(path, image_shape)
        saver = VideoSaver(
            save_dir,
            loader.original_fps,
            loader.original_frame_size,
            target_fps,
        )
        return loader, saver, True
    # webcam
    if path.isnumeric():
        loader = WebcamLoader(int(path), image_shape)
        saver = (
            VideoSaver(save_dir, 30, loader.original_frame_size, None)
            if not no_save
            else None
        )
        return loader, saver, True
    # image file(s)
    return ImageLoader(path, image_shape, batch_size), ImageSaver(save_dir), False

class ImageSaver:
    """
    Base class for saving YOLO model outputs. Saves each image as an individual file in
    the given directory

    :param save_dir: path to directory to write to
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._idx = 0

        create_dirs(save_dir)

    def save_frame(self, image: numpy.ndarray):
        """
        :param image: numpy array of image to save
        """
        output_path = os.path.join(self._save_dir, f"result-{self._idx}.jpg")
        cv2.imwrite(output_path, image)
        self._idx += 1

    def close(self):
        """
        perform any clean-up tasks
        """
        pass

class VideoLoader:
    """
    Class for pre-processing and iterating over video frames to be used as input for
    YOLO models

    :param path: Filepath to single video file
    :param image_size: size of input image_batch to model
    """

    def __init__(self, path: str, image_size: Tuple[int, int] = (640, 640)):
        self._path = path
        self._image_size = image_size
        self._vid = cv2.VideoCapture(self._path)
        self._total_frames = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._vid.get(cv2.CAP_PROP_FPS)

    def __iter__(self) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        for _ in range(self._total_frames):
            loaded, frame = self._vid.read()
            if not loaded:
                break
            yield load_image(frame, image_size=self._image_size)
        self._vid.release()

    def __len__(self):
        return len(self._total_frames)

    @property
    def original_fps(self) -> float:
        """
        :return: the frames per second of the video this object reads
        """
        return self._fps

    @property
    def original_frame_size(self) -> Tuple[int, int]:
        """
        :return: the original size of frames in the video this object reads
        """
        return (
            int(self._vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def total_frames(self) -> int:
        """
        :return: the total number of frames this object may laod from the video
        """
        return self._total_frames

class ImageLoader:
    """
    Class for pre-processing and iterating over image_batch to be used as input for YOLO
    models

    :param path: Filepath to single image file or directory of image files to load,
        glob paths also valid
    :param image_size: size of input image_batch to model
    """

    def __init__(self, path: str, image_size: Tuple[int, int] = (640, 640), batch_size = 1):
        self._path = path
        self._image_size = image_size
        self._batch_size = batch_size

        if os.path.isdir(path):
            self._image_file_paths = [
                os.path.join(path, file_name) for file_name in os.listdir(path)
            ]
        elif "*" in path:
            self._image_file_paths = glob.glob(path)
        elif os.path.isfile(path):
            # single file
            if self._batch_size != 1:
                raise ValueError("Attempting to instantiate `ImageLoader` that holds a "
                                 f"single image, but the `batch_size` is set to {self.batch_size}. "
                                 "Set `batch_size = 1`")
            self._image_file_paths = [path]
        else:
            raise ValueError(f"{path} is not a file, glob, or directory")

    def __iter__(self) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:

        for image_path in self._image_file_paths:
            batch_input, batch_src = [], []
            for _ in range(self._batch_size):
                input_img, src_img = load_image(image_path, image_size=self._image_size)
                batch_input.append(input_img)
                batch_src.append(src_img)
            yield numpy.stack(batch_input), numpy.stack(batch_src)


    def __len__(self):
        return len(self._image_file_paths)

class VideoSaver(ImageSaver):
    """
    Class for saving YOLO model outputs as a VideoFile

    :param save_dir: path to directory to write to
    :param original_fps: frames per second to save video with
    :param output_frame_size: size of frames to write
    :param target_fps: fps target for output video. if present, video
        will be written with a certain number of the original frames
        evenly dropped to match the target FPS.
    """

    def __init__(
        self,
        save_dir: str,
        original_fps: float,
        output_frame_size: Tuple[int, int] = (640, 640),
        target_fps: Optional[float] = None,
    ):
        super().__init__(save_dir)

        self._output_frame_size = (640,640)
        self._original_fps = original_fps

        if target_fps is not None and target_fps >= original_fps:
            print(
                f"target_fps {target_fps} is greater than source_fps "
                f"{original_fps}. target fps file will not be invoked"
            )
        self._target_fps = target_fps

        self._file_path = os.path.join(self._save_dir, "results.mp4")
        self._writer = cv2.VideoWriter(
            self._file_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            original_fps,
            self._output_frame_size,
        )
        self._n_frames = 0

    def save_frame(self, image: numpy.ndarray):
        """
        :param image: numpy array of image to save
        """
        self._writer.write(image)
        self._n_frames += 1

    def close(self):
        """
        perform any clean-up tasks
        """
        self._writer.release()
        if self._target_fps is not None and self._target_fps < self._original_fps:
            self._write_target_fps_video()

    def _write_target_fps_video(self):
        assert self._target_fps is not None
        num_frames_to_keep = int(
            self._n_frames * (self._target_fps / self._original_fps)
        )
        # adjust target fps so we can keep the same video duration
        adjusted_target_fps = num_frames_to_keep * (self._original_fps / self._n_frames)

        # select num_frames_to_keep evenly spaced frame idxs
        frame_idxs_to_keep = set(
            numpy.round(numpy.linspace(0, self._n_frames, num_frames_to_keep))
            .astype(int)
            .tolist()
        )

        # create new video writer for adjusted video
        vid_path = os.path.join(
            self._save_dir, f"_results-{adjusted_target_fps:.2f}fps.mp4"
        )
        fps_writer = cv2.VideoWriter(
            vid_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            adjusted_target_fps,
            self._output_frame_size,
        )

        # read from original video and write to FPS adjusted video
        saved_vid = cv2.VideoCapture(self._file_path)
        for idx in range(self._n_frames):
            _, frame = saved_vid.read()
            if idx in frame_idxs_to_keep:
                fps_writer.write(frame)

        saved_vid.release()
        fps_writer.release()
        shutil.move(vid_path, self._file_path)  # overwrite original file


def load_image(
    img: Union[str, numpy.ndarray], image_size: Tuple[int, int] = (640, 640)
) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
    """
    :param img: file path to image or raw image array
    :param image_size: target shape for image
    :return: Image loaded into numpy and reshaped to the given shape and the original
        image
    """
    img = cv2.imread(img) if isinstance(img, str) else img
    img_resized = cv2.resize(img, (224, 224))
    img_transposed = img_resized[:, :, ::-1].transpose(2, 0, 1)

    return img_transposed, img


def get_annotations_save_dir(
    initial_save_dir: str,
    tag: Optional[str] = None,
    engine: Optional[str] = None,
) -> str:
    """
    Returns the directory to save annotations to. If directory exists and is
    non-empty, a number is appended to the end of the directory name.

    :param initial_save_dir: Initial directory to save annotations to
    :param tag: A tag under which to save the annotations inside `save_dir`
    :param engine: Used to generate a unique tag if it is not provided.
    :return: A new unique dir path to save annotations to
    """
    name = tag or f"{engine}-annotations"
    initial_save_dir = os.path.join(initial_save_dir, name)
    counter = 0
    new_save_dir = initial_save_dir
    while Path(new_save_dir).exists() and any(Path(new_save_dir).iterdir()):
        counter += 1
        new_save_dir = os.path.join(initial_save_dir, f"{name}-{counter:03d}")

    _LOGGER.info(f"Results will be saved to {new_save_dir}")
    Path(new_save_dir).mkdir(parents=True, exist_ok=True)
    return new_save_dir

def annotate(
    pipeline: "YOLOPipeline",  # noqa: F821
    annotation_func,
    image_batch: Union[List[numpy.ndarray], List[str]],
    target_fps: float = None,
    calc_fps: bool = False,
    original_images: Optional[Union[List[numpy.ndarray], numpy.ndarray]] = None,
    **kwargs,
) -> List[numpy.ndarray]:
    """
    Annotate and return image_batch with bounding boxes and labels

    :param pipeline: A YOLOPipeline object
    :param image_batch: A list of image files, or batch of numpy image_batch
    :param target_fps: If not None, then the pipeline will be run at this target
    :param calc_fps: If True, and target_fps is None then the pipeline will
        calculate the FPS
    :param original_images: images from input_batch before any processing
    :return: A list of annotated images

    """

    if not isinstance(image_batch, list):
        if isinstance(image_batch, numpy.ndarray) and image_batch.shape == 4:
            image_batch = list(image_batch)
        else:
            image_batch = [image_batch]

    if not original_images:
        original_images = image_batch

    batch_size = len(image_batch)
    single_batch = batch_size == 1
    if image_batch and isinstance(image_batch[0], str):
        original_images = [cv2.imread(image) for image in image_batch]

    if target_fps is None and calc_fps:
        start = time.time()
    image_batch = numpy.stack(image_batch)
    pipeline_outputs = pipeline(images=image_batch)

    if target_fps is None and calc_fps:
        target_fps = float(batch_size) / (time.time() - start)

    annotated_images = []
    if single_batch:
        pipeline_outputs = [pipeline_outputs]

    for index, image_output in enumerate(pipeline_outputs):
        image = original_images[index]
        result = annotation_func(image, image_output, **kwargs)
        annotated_images.append(result)

    return annotated_images

class WebcamLoader:
    """
    Class for pre-processing and iterating over webcam frames to be used as input for
    YOLO models.

    Adapted from: https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py

    :param camera: Webcam index
    :param image_size: size of input image_batch to model
    """

    def __init__(self, camera: int, image_size: Tuple[int, int] = (640, 640)):

        self._camera = camera
        self._image_size = image_size
        self._stream = cv2.VideoCapture(self._camera)
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def __iter__(self) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        while True:
            if cv2.waitKey(1) == ord("q"):  # q to quit
                self._stream.release()
                cv2.destroyAllWindows()
                break
            loaded, frame = self._stream.read()

            assert loaded, f"Could not load image from webcam {self._camera}"

            frame = cv2.flip(frame, 1)  # flip left-right
            yield load_image(frame, image_size=self._image_size)

    @property
    def original_frame_size(self) -> Tuple[int, int]:
        """
        :return: the original size of frames in the stream this object reads
        """
        return (
            int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )