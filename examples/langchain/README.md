
## Install libraries

`pip install -r requirements.txt`

### Run the server
```
deepsparse.server \
  --task sentiment-analysis \
  --model_path zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```


# Run langchain
`python main.py`
