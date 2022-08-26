// Damian: this is the draft of the blog post, stripped down from images, most of the hyperlinks and "commercial jargon", that we should probably insert at some place (like pointers to deepsparse or mentioning what we do etc.).  I would leave the beautification and aesthetic part for later. You will find also some other random comments from me throughout the writeup. Also the names of paragraphs are more informatory for the reviewers, then final.

## The Use Case

Accurately detecting and segmenting objects during the sorting and packaging helps not only to improve the quality of the process but may also significantly lower inspection costs of outbound packages. 

Fruit detection may serve as an example of such a task. Here the objective is to identify the presence of fruit, classify it, and finally accurately locate it. 

Instance segmentation might be additionally used to provide detailed information about the location of the object. This may potentially enable building autonomous pick-and-place systems, in which robots are deployed to manipulate (sort or pack) the object of interest. It is the detailed information about the object's geometry, captured by instance segmentation, that may enable a machine to figure out how to grasp fruit with its robotic fingers and then successfully place it in a bin.

## Neural Magic!

But the logistics processes may be quite messy and difficult to automate. It is not enough just to deploy a network, even one that delivers high-quality detection. Many engineers would be surprised by the possible set of constraints such as a need for high latency, limited available compute resources, or limited connectivity in the facility.

What do most decision makers do when they want to deploy Artificial Intelligence (AI) solutions? They need a lot of compute, so they either invest in a huge GPU cluster or rent one from one of the dozens of cloud providers.

One may stipulate that this approach is frequently not realistic for many businesses. For a sorting task, we require a reliable system that can rapidly and continuously run inference: 24 hours a day, 7 days a week. This means that there is a significant risk that using the cloud provider may not fulfill these requirements. Especially given the fact that in our sorting facility we might very often encounter areas of limited connectivity.

So let us buy our very own GPU cluster then! Well, procuring and maintaining the cluster sounds like a big financial investment. Why go down that path when practically every sorting facility surely is equipped with dozens of computers using good old commodity CPUs? The CPUs are already set up (no need for financial commitment), wired to the facility system (widely available), and already interfacing with all the machinery (the required latency is most likely secured).

Our team in Neural Magic delivers just the right solution for such a scenario - the neural network inference engine that delivers GPU-class performance on commodity CPUs! The DeepSparse Engine is taking advantage of the sparsified (compressed) neural networks to deliver state-of-the-art inference speed. The engine is particularly effective when running network sparsified using methods such as pruning and quantization. These techniques result in significantly more performant and smaller models with limited to no effect on the baseline metrics.

In this post, we show you how to effectively deploy an instance segmentation model, using a standard quad-core CPU processor. We choose the task of fruit segmentation to showcase the effectiveness and speed of our solution, and potentially convince several sorting facility owners that automation through AI brings massive value.


## What is YOLACT

The model, that we will be used in this write-up is [YOLACT](https://arxiv.org/abs/1904.02689) [Boyla et al. 2019]. YOLACT stands for "You Only Look At CoefficienTs". 
YOLACT was one of the first methods, that was able to do instance segmentation in real-time. 

YOLACT is an extension of the popular YOLO (You Only Look Once algorithm, check out our Neural Magic tutorial on YOLO here: ...). YOLO is one of the first deep learning algorithms for real-time object detection. It achieves its speed by forgoing the conventional (and slow) two-stage detection process: generating regions of proposals and then classifying those regions. The two-stage process is used by many conventional detection networks such as Mask R-CNN. YOLO, on the other hand, directly predicts a fixed number of bounding boxes on the image. This is being done in a single step and therefore, is fast!

YOLACT is an extension of YOLO for instance segmentation. Similarly to its predecessor, YOLACT also forgoes the step of explicit localization of regions of proposals. Instead, it breaks up instance segmentation into two parallel tasks: generating a dictionary of prototype masks over the entire image and predicting a set of linear combination coefficients per detection. This not only produces a full-image instance segmentation mask (from these two components), but it also works very fast!

In the context of this tutorial, this is the most important feature of YOLACT: speed. Since our fruit sorting task needs to work in real-time, YOLACT is an adequate choice.

## Running Instance Segmentation using the DeepSparse Engine

The model that we are deploying comes from the [Sparsezoo](http://sparsezoo.neuralmagic.com) - the Neural Magic repository of sparsified, pre-trained neural networks. The particular model used in this write-up is a [Pruned82 Quant](https://sparsezoo.neuralmagic.com/models/cv%2Fsegmentation%2Fyolact-darknet53%2Fpytorch%2Fdbolya%2Fcoco%2Fpruned82_quant-none) model. This means that 82 percent of the model's weights are removed, and the rest of the parameters are quantized. 

In this tutorial, we are benchmarking the sparsified model against its dense counterpart. Both models are pre-trained on the [COCO dataset](https://cocodataset.org). While the original, dense model is deployed in the ONNXRuntime engine, the sparse model is run in the Neural Magic's DeepSparse engine, to fully take advantage of the model's sparsity.

Below you find a short comparison of two models that are used for this tutorial:


| Model type 	| Pruned [%] 	| Quantized 	| mAP@all 	| Size [mb] 	| trained on 	| inference engine 	|
|:----------:	|:----------:	|:---------:	|:-------:	|:---------:	|:----------:	|:----------------:	|
| Sparsified 	|     82     	|    yes    	|   28.2  	|    9.7    	|    COCO    	|    DeepSparse    	|
|    Dense   	|      0     	|     no    	|   28.8  	|   170.0   	|    COCO    	|    ONNXRuntime   	|


Notice the difference in size (compressed, on the disk) of those two models. The sparsified model is smaller than the dense model by more than an order of magnitude. However, this does not noticeably impact the quality of the model - expressed in terms of the mAP (mean Average Precision) metric.

### Setup
// Damian: All the numbers/media I am displaying here come from running the models on the benchmarking server with 4 cores. Is there anything else that we should mention to the readers other than the amount of cores?


### Examples

//Damian: here we will be showing videos/media

//Damian: this can be a funny capture next to the videos

The orange gets sometimes classified as an apple, and even a doughnut! Pretty significant detail for those who are trying to stay fit!

//Damian: this is the summary of what the user might have learned from the videos

The obvious thing that might be spotted instantaneously is the speed gain that comes from deploying the sparsified model in the DeepSparse engine. On average, we see that the inference is 3 times faster than in case of its dense counterpart. The sparsified model is fast enough to support real-time inference. This is great: a real-time instance segmentation model that runs exclusively on CPU!

As expected - removing (big amounts of) redundant information from the overprecise and over-parameterized network results in a faster and smaller model, while practically retaining the original quality. 

Additionally, we may see that the accuracy of the sparsified model is indeed on par with the dense model. That's remarkable. Even though we are keeping only two out of ten original neurons of the network (plus we are "dumbing them down" through quantization), there is no noticeable drop in the quality of the inference. 

Nevertheless, the reader may note that when it comes to the inference quality of both networks (independently of the sparsity degree), there still remains some room for improvement. This is largely due to the fact, that models pre-trained on COCO dataset are fundamentallly not adequate to be deployed "in-the-wild". The data distribution observed in our fruit detection use case is very different from the COCO data distribution. Our main goal of the write-up is to demonstrate the inference speed gain, not the quality of inference. Having said that, we will be soon enabling the users to access the sparsified pre-trained networks and "sparse transfer learn" on private datasets. That will solve the problem in question, and enable high quality, fast CPU inference in different scenarios.







