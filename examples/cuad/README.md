# CUAD

CUAD is a legal question answering dataset, which means you can use our built in question-answering pipeline
to answer questions given a model.

For an example, see the included `main.py` file.

Running this on a model fine tuned on the cuad dataset outputs something like the following:

```bash
$ python examples/cuad/main.py <path to model deployent dir>
...
Answered: SUPPLY CONTRACT Contract No: Date: The buyer/End-User: Shenzhen LOHAS Supply Chain Management Co., Ltd. ADD: Tel No. : Fax No. : The seller: ADD: The Contract is concluded and signed by the Buyer and Seller on , in Hong Kong.
Expected: {'text': ['SUPPLY CONTRACT'], 'answer_start': [14]}
```
