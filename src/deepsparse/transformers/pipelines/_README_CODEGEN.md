# Engineering Log for Codegen project, could be later developed into a README

## Export

## SparseML export

### In Sparseml
- checkout the branch `feature/damian/export_text_generation`
- run:

```bash
git lfs install
git clone https://huggingface.co/Salesforce/codegen-350M-multi
```

- run `sparseml.transformers.export --model_path codegen-350M-multi --task codegen`

The resulting directory `deployment ` can be passed to the pipeline to start inference.

## Hugging Face Export

### Clone The Model Repo
```bash
git lfs install
git clone https://huggingface.co/Salesforce/codegen-350M-multi
```

### Export the ONNX model
```bash
pip install "transformers[torch,onnx]" 
python -m transformers.onnx -m Salesforce/codegen-350M-multi codegen-350M-multi --atol=5e-5 --feature causal-lm
```
- `atol` argument was neccessary for me. Otherwise you may run into the following issue:
```bash
ValueError: Outputs values doesn't match between reference model and ONNX exported model: Got max absolute difference of: 2.002716064453125e-05 for [ 0.08507037  0.15080063  0.0017249  ...  0.16075289 -0.14426014
  0.26479816] vs [ 0.08506441  0.15079397  0.00173104 ...  0.16076398 -0.1442534
  0.2648096 ]
```
- `feature` argument ensures that the network outputs logits

The resulting directory `codegen-350M-multi` can be passed to the pipeline to start inference.


## Quantization
Quantize either model use
`python src/deepsparse/transformers/codegen_utils/quantize.py --model {ONNX_MODEL_NAME}`



