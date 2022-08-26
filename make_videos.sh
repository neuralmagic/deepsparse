rm -rf /home/damian/deepsparse/annotation-results # just to keep our results always clean (edit it)
VIDEO_DIRECTORY=/home/damian/fruity_videos/* # path to your 'fruity_videos' (edit it)

# MEAN_FPS_DEEPSPARSE for the fruity videos is about 13 FPS (benchmarking server, sparse model)
# MEAN_FPS_ONNXRUNTIME for the fruity videos is about 3 3 (benchmarking server, dense model)
STUB=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none

for FILE in $VIDEO_DIRECTORY;
    do echo Running inference deepsparse engine using $STUB on $FILE;
    deepsparse.instance_segmentation.annotate --num_cores 4 --source $FILE --model-filepath $STUB;
    done;

STUB=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none
for FILE in $VIDEO_DIRECTORY;
    do echo Running inference onnxruntime engine using $STUB on $FILE;
    deepsparse.instance_segmentation.annotate --num_cores 4 --source $FILE --engine onnxruntime --model-filepath $STUB;
    done;
