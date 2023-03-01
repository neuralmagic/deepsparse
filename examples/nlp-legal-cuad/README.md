# CUAD

CUAD is a legal question answering dataset, which means you can use our built in question-answering pipeline
to answer questions given a model.

For an example, see the included `main.py` file.

Running this on a model fine tuned on the cuad dataset outputs something like the following:

```bash
$ python examples/cuad/main.py <path to model deployent dir>
...
Answered: SUPPLY CONTRACT Contract
Expected: {'text': ['SUPPLY CONTRACT'], 'answer_start': [14]}
```

where the question was:
```
Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer. Details: The name of the contract
```

and the (trimmed because its very long) context is:

```
Context Exhibit 10.16 SUPPLY CONTRACT Contract No: Date: The buyer/End-User: Shenzhen LOHAS Supply Chain Management Co., Ltd. ADD: Tel No. : Fax No. : The seller: ADD: The Contract is concluded and signed by the Buyer and Seller on , in Hong Kong. 1. General provisions 1.1 This is a framework agreement, the terms and conditions are applied to all purchase orders which signed by this agreement (hereinafter referred to as the "order"). 1.2 If the provisions of the agreement are inconsistent with the order, the order shall prevail. Not stated in order content will be subject to the provisions of agreement. Any modification, supplementary, give up should been written records, only to be valid by buyers and sellers authorized representative signature and confirmation, otherwise will be deemed invalid. 2. The agreement and order 2.1 During the validity term of this agreement, The buyer entrust SHENZHEN YICHANGTAI IMPORT AND EXPORT TRADE CO., LTD or SHENZHEN LEHEYUAN TRADING CO, LTD (hereinafter referred to as the "entrusted party" or "YICHANGTAI" or "LEHEYUAN"), to purchase the products specified in this agreement from the seller in the form of orders. 2.2 The seller shall be confirmed within three working days after receipt of order. If the seller finds order is not acceptable or need to modify, should note entrusted party in two working days after receipt of the order, If the seller did not confirm orders in time or notice not accept orders or modifications, the seller is deemed to have been accepted the order. The orders become effective once the seller accepts, any party shall not unilaterally cancel the order before the two sides agreed . 2.3 If the seller puts forward amendments or not accept orders, the seller shall be in the form of a written notice to entrusted party, entrusted party accept the modified by written consent, the modified orders to be taken effect. 2.4 Seller's note, only the buyer entrust the entrusted party issued orders, the product delivery and payment has the force of law.
...
<snip>
```