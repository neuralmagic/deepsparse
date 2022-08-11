rm -rf /home/damian/deepsparse_copy/annotation-results

VIDEO_DIRECTORY=/home/damian/deepsparse_copy/src/videos/*
STUB=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none
FPS=5

for FILE in $VIDEO_DIRECTORY;
    do echo Running inference using $STUB on $FILE;
    deepsparse.instance_segmentation.annotate --source $FILE --target-fps $FPS --model-filepath $STUB;
    done;

STUB=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none
FPS=20

for FILE in $VIDEO_DIRECTORY;
    do echo Running inference using $STUB on $FILE;
    deepsparse.instance_segmentation.annotate --source $FILE --target-fps $FPS --model-filepath $STUB;
    done;

STUB=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none

for FILE in $VIDEO_DIRECTORY;
    do echo Running inference using $STUB on $FILE;
    deepsparse.instance_segmentation.annotate --source $FILE --engine onnxruntime --model-filepath $STUB;
    done;
