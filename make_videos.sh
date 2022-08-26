rm -rf /home/damian/deepsparse_copy/annotation-results

VIDEO_DIRECTORY=/home/damian/deepsparse_copy/src/fruity_videos/* # path to your 'fruity_videos'
MEAN_FPS_DEEPSPARSE=0
MEAN_FPS_ONNXRUNTIME=0
STUB=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none

for FILE in $VIDEO_DIRECTORY;
    do echo Running inference deepsparse engine using $STUB on $FILE;
    deepsparse.instance_segmentation.annotate --source $FILE --model-filepath $STUB;
    done;

STUB=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none
for FILE in $VIDEO_DIRECTORY;
    do echo Running inference onnxruntime engine using $STUB on $FILE;
    deepsparse.instance_segmentation.annotate --source $FILE --engine onnxruntime --model-filepath $STUB;
    done;
