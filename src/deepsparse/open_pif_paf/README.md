# OpenPifPaf Inference Pipelines

The DeepSparse integration of the OpenPifPaf model is a work in progress. Please check back soon for updates.
This README serves as a placeholder for internal information that may be useful for further development.

DeepSparse pipeline for OpenPifPaf

## Example use in DeepSparse Python API:

```python
from deepsparse import Pipeline

model_path: str = ... # path to open_pif_paf model (SparseZoo stub or onnx model)
pipeline = Pipeline.create(task="open_pif_paf", model_path=model_path)
predictions = pipeline(images=['dancers.jpg'])
# predictions have attributes `data', 'keypoints', 'scores', 'skeletons'
predictions[0].scores
>> scores=[0.8542259724243828, 0.7930507659912109]
```
### Output CifCaf fields
Alternatively, instead of returning the detected poses, it is possible to return the intermediate output - the CifCaf fields.
This is the representation returned directly by the neural network, but not yet processed by the matching algorithm

```python
...
pipeline = Pipeline.create(task="open_pif_paf", model_path=model_path,  return_cifcaf_fields=True)
predictions = pipeline(images=['dancers.jpg'])
predictions.fields
```

## Validation script:
This paragraph describes how to run validation of the ONNX model/SparseZoo stub

### Dataset
For evaluation, you need to download the dataset. The [Open Pif Paf documentation](https://openpifpaf.github.io/) describes 
thoroughly how to prepare different datasets for validation. This is the example for `crowdpose` dataset:

```bash
mkdir data-crowdpose
cd data-crowdpose
# download links here: https://github.com/Jeff-sjtu/CrowdPose
unzip annotations.zip
unzip images.zip
# Now you can use the standard openpifpaf.train and openpifpaf.eval 
# commands as documented in Training with --dataset=crowdpose.
```

### Validation command
Once the dataset has been downloaded, run the command:
...

## The necessity of external OpenPifPaf helper function 
<img width="678" alt="image" src="https://user-images.githubusercontent.com/97082108/203295520-42fa325f-8a94-4241-af6f-75938ef26b14.png">

As illustrated by the diagram from the original paper: once the input image has been encoded into PIF or PAF (or CIF and CAF, same meaning, just two different naming conventions 
per original authors) tensors, they need to be later decoded into human-understandable annotations.

Once the neural network outputs CIF and CAF tensors, they are then processed by an algorithm described below:

<img width="337" alt="image" src="https://user-images.githubusercontent.com/97082108/203295686-91305e9c-e455-4ac8-9652-978f9ec8463d.png">

For speed reasons, the decoding in the original `OpenPifPaf` repository is implemented in [C++ and libtorch](https://github.com/openpifpaf/openpifpaf/issues/560): `https://github.com/openpifpaf/openpifpaf/src/openpifpaf/csrc`

Rewriting this functionality would be a significant engineering effort, so I reuse part of the original implementation in the pipeline:

### On the pipeline instantiation

```python
model_cpu, _ = network.Factory().factory(head_metas=None)
self.processor = decoder.factory(model_cpu.head_metas)
```

First, I fetch the default `model` object (also a second argument, which is the last epoch of the pre-trained model) from the factory. Note, this `model` will not be used for inference, only to pull the information 
about the heads of the model: `model_cpu.head_metas: List[Cif, Caf]`. This information will be consumed to create a (set of) decoder(s) (objects that map `fields`, raw network output, to human-understandable annotations).

Note: The `Cif` and `Caf` objects seem to be dataset-dependent. They hold e.g. the information about the expected relationship of the joints of the pose (skeleton).

Hint: Instead of returning Annotation objects, the API supports returning annotations as JSON serializable dicts. This is probably what we should aim for.

In the default scenario (I suspect for all the pose estimation tasks), the `self.processor` will be a `Multi` object that holds a single `CifCaf` decoder. 

Other available decoders:

```python
{
openpifpaf.decoder.cifcaf.CifCaf,
openpifpaf.decoder.cifcaf.CifCafDense, # not sure what this does
openpifpaf.decoder.cifdet.CifDet, # I think this is just for the object detection task
openpifpaf.decoder.pose_similarity.PoseSimilarity, # for pose similarity task
openpifpaf.decoder.tracking_pose.TrackingPose # for tracking task
}
```

## On the engine output preprocessing 

```python
 def process_engine_outputs(self, fields):
 
     for idx, (cif, caf) in enumerate(zip(*fields)):
         annotations = self.processor._mappable_annotations(
                [torch.tensor(cif), 
                 torch.tensor(caf)], None, None)
```
I am passing the CIF and CAF values directly to the processor (through the private function; `self.processor` itself, by default, does batching and inference). 
This is the functionality that we perhaps would like to fold into our computational graph:
1. To avoid being dependent on the external library and their torch-dependent implementation
2. Having control (and the possibility to improve upon) over the generic decoder






