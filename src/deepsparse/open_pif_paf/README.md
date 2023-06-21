# OpenPifPaf Inference Pipelines

The DeepSparse integration of the OpenPifPaf model is a work in progress. Check back soon for updates.
This README serves as a placeholder for internal information that may be useful for further development.

DeepSparse pipeline for OpenPifPaf

## Example Use in DeepSparse Python API

```python
from deepsparse import Pipeline

model_path: str = ... # path to open_pif_paf model (SparseZoo stub or onnx model)
pipeline = Pipeline.create(task="open_pif_paf", model_path=model_path)
predictions = pipeline(images=['dancers.jpg'])
# predictions have attributes `data', 'keypoints', 'scores', 'skeletons'
predictions[0].scores
>> scores=[0.8542259724243828, 0.7930507659912109]
```
### Output CifCaf Fields
Alternatively, instead of returning the detected poses, it is possible to return the intermediate output&mdash;the CifCaf fields.
This is the representation returned directly by the neural network, but not yet processed by the matching algorithm.

```python
...
pipeline = Pipeline.create(task="open_pif_paf", model_path=model_path,  return_cifcaf_fields=True)
predictions = pipeline(images=['dancers.jpg'])
predictions.fields
```

## Validation Script
This section describes how to run validation of the ONNX model/SparseZoo stub.

### Dataset
For evaluation, you need to download the dataset. The [Open Pif Paf documentation](https://openpifpaf.github.io/) 
thoroughly describes how to prepare different datasets for validation. This is the example for the `crowdpose` dataset:

```bash
mkdir data-crowdpose
cd data-crowdpose
# download links here: https://github.com/Jeff-sjtu/CrowdPose
unzip annotations.zip
unzip images.zip
# Now you can use the standard openpifpaf.train and openpifpaf.eval 
# commands as documented in Training with --dataset=crowdpose.
```
### Create an ONNX Model

```bash
python3 -m openpifpaf.export_onnx --input-width 641 --input-height 641
```

### Validation Command
Once the dataset has been downloaded, run the command:
```bash
deepsparse.pose_estimation.eval --model-path openpifpaf-resnet50.onnx  --dataset cocokp --image_size 641
```

This should result in the evaluation output similar to this:
```bash
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.502
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.732
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.523
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.429
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.605
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.744
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.643
...
````


### Expected Output

## Necessity of the External OpenPifPaf Helper Function 

This diagram from the original paper illustrates that once the input image has been encoded into PIF or PAF* 
tensors, they need to be decoded later into human-understandable annotations. (* PIF and PAF are equivalent to CIF and CAF with the same meaning but different naming conventions per the original authors.)

<img width="678" alt="image" src="https://user-images.githubusercontent.com/97082108/203295520-42fa325f-8a94-4241-af6f-75938ef26b14.png">

Once the neural network outputs PIF and PAF tensors, they are processed by an algorithm described below:

<img width="337" alt="image" src="https://user-images.githubusercontent.com/97082108/203295686-91305e9c-e455-4ac8-9652-978f9ec8463d.png">

For speed reasons, the decoding in the original `OpenPifPaf` repository is implemented in [C++ and libtorch](https://github.com/openpifpaf/openpifpaf/issues/560): `https://github.com/openpifpaf/openpifpaf/src/openpifpaf/csrc`

Rewriting this functionality would be a significant engineering effort, so we reuse part of the original implementation in the pipeline, as shown below.

### Pipeline Instantiation

```python
model_cpu, _ = network.Factory().factory(head_metas=None)
self.processor = decoder.factory(model_cpu.head_metas)
```

First, we fetch the default `model` object (also a second argument, which is the last epoch of the pre-trained model) from the factory. Note, this `model` will not be used for inference, but rather to pull the information 
about the heads of the model: `model_cpu.head_metas: List[Cif, Caf]`. This information will be consumed to create a (set of) decoder(s) (objects that map `fields`, raw network output, to human-understandable annotations).

Note: The `Cif` and `Caf` objects seem to be dataset-dependent. For example, they hold the information about the expected relationship of the joints of the pose (skeleton).

Hint: Instead of returning Annotation objects, the API supports returning annotations as JSON serializable dicts. This is probably what we should aim for.

In the default scenario (likely for all the pose estimation tasks), the `self.processor` will be a `Multi` object that holds a single `CifCaf` decoder. 

Other available decoders are:

```python
{
openpifpaf.decoder.cifcaf.CifCaf,
openpifpaf.decoder.cifcaf.CifCafDense, # not sure what this does
openpifpaf.decoder.cifdet.CifDet, # I think this is just for the object detection task
openpifpaf.decoder.pose_similarity.PoseSimilarity, # for pose similarity task
openpifpaf.decoder.tracking_pose.TrackingPose # for tracking task
}
```

## Engine Output Preprocessing 

```python
 def process_engine_outputs(self, fields):
 
     for idx, (cif, caf) in enumerate(zip(*fields)):
         annotations = self.processor._mappable_annotations(
                [torch.tensor(cif), 
                 torch.tensor(caf)], None, None)
```
We are passing the CIF and CAF values directly to the processor through the private function (`self.processor`, by default, does batching and inference). 
Perhaps this is the functionality that we would like to fold into our computational graph:
1. To avoid being dependent on the external library and their torch-dependent implementation
2. Having control over (and the possibility to improve upon) the generic decoder






