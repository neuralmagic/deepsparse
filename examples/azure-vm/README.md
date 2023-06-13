<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# **Getting Started With DeepSparse in an Azure VM**

Neural Magic’s DeepSparse is an inference runtime that can be deployed directly from a public Docker image. DeepSparse supports various CPU instance types and sizes, allowing you to quickly deploy the infrastructure that works best for your use case, based on cost and performance.

If you are interested in configuring and launching an instance with DeepSparse in Python, follow the step-by-step guide below. 

We recommend installing the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) for easy access to Azure's functionalities although it is not necessary.

## Step 1: Create a Subscription
[Create an Azure subscription](https://learn.microsoft.com/en-us/azure/cost-management-billing/manage/create-subscription) to gain access to a `subscription id`.


## Step 2: Install Dependencies

```bash
git clone https://github.com/neuralmagic/deepsparse.git
cd deepsparse/examples/azure-vm
pip install -r requirements.txt
```

## Step 3: Run Script

This [azure-vm.py script](https://github.com/neuralmagic/deepsparse/tree/main/examples/azure-vm/azure-vm.py) creates an Azure resource group, launches an Ubuntu instance and returns the Public IP address so you can SSH into the instance after it finishes staging. Additionally, it also contains a bash script that automatically downloads Docker and pulls Neural Magic's public DeepSparse image into your instance.

To execute the script, run the following command and pass in your `subscription id` from step 1, your VMs `location`, `vm-type`, a resources `group name`, your `virtual machine's name` and the `password` to login into your instance:

```bash
python azure-vm.py create-vm --subscription-id <SUBSCRIPTION-ID> --location <LOCATION> --vm-type <VM-TYPE> --group-name <GROUP-NAME> --vm-name <VM-NAME> --pw <PASSWORD>
```

To leverage CPU optimized instances, we recommend using the [`Fsv2-series`](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-compute) instances which contain AVX-512 instructions. Here's an example commmand for launching a VM in the US East location using F4s-v2 instance containing 4 vCPUs and 8GB of RAM: 

```bash
python azure-vm.py create-vm --subscription-id <sub-id> --location eastus --vm-type Standard_F4s_v2 --group-name deepsparse-group --vm-name deepsparse-vm --pw Password123!
```

**PRO-TIP**: The password passed into the CLI command must satisfy the following conditions:

1) Contains an uppercase character.
2) Contains a lowercase character.
3) Contains a numeric digit.
4) Contains a special character.
5) Control characters are not allowed.

## **Step 4: SSH Into the Instance**

After running the script, your instance's public IP address will be printed out in the terminal. Pass the IP address into the following CLI command to SSH into your running instance:

```bash
ssh testuser@<PUBLIC-IP>
```

Get root access:

After entering your password, get root:

```bash
sudo su
```

## **Step 5: Run DeepSparse**

We recommend giving your instance 2-3 mins. to finish executing the bash script. To make sure you have the DeepSparse image imported, run the following command:

```bash
docker images
```
You should be able to see DeepSparse image installed, if it shows an empty table, the bash script hasn't completed execution.

After you see the image in your instance, you can now execute DeepSparse from the running container. Here's an example of running the DeepSparse server a pruned-quantized version of BERT trained on SQuAD:

```bash
docker run -it deepsparse_docker deepsparse.benchmark zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none -i [64,128] -b 64 -nstreams 1 -s sync
```

## **Step 6: Delete Instance and Resource Group**

```bash
python azure-vm.py delete-vm-rg --subscription-id <SUBSCRIPTION-ID> --group-name <GROUP-NAME> --vm-name <VM-NAME>
```