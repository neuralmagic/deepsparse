For training and evaluation, you need to download the dataset.

mkdir data-crowdpose
cd data-crowdpose
# download links here: https://github.com/Jeff-sjtu/CrowdPose
unzip annotations.zip
unzip images.zip
Now you can use the standard openpifpaf.train and openpifpaf.eval commands as documented in Training with --dataset=crowdpose.

install pycocotools