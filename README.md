# [Related Icon Here] The Engine

### Inference engine for running neural networks efficiently and performantly

<p>
    <a href="https://github.com/neuralmagic/comingsoon/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/comingsoon.svg?color=purple&style=for-the-badge" height=25>
    </a>
    <a href="https://docs.neuralmagic.com/engine/">
        <img alt="Documentation" src="https://img.shields.io/website/http/neuralmagic.com/engine/index.html.svg?down_color=red&down_message=offline&up_message=online&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/engine/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/engine.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic.com/comingsoon/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25>
    </a>
     <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
        <img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=25>
    </a>
     <a href="https://medium.com/limitlessai">
        <img src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height=25>
    </a>
    <a href="https://twitter.com/neuralmagic">
        <img src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height=25>
    </a>
 </p>

## Overview

Neural Magic, the creators of The Engine, is focused on making performance engineering for deep learning easy, affordable, and accessible. The Neural Magic Inference Engine enhances speed for neural networks in numerous ways, including activation sparsity and model pruning. End users can run mission critical deep learning models on commodity CPUs to reduce cost per inferences and generate price-performant deployments. This feature set includes the inference engine, ONNX conversion tooling, model server if needed, and is focused on model deployment and scaling machine learning pipelines.

This repository contains information on how to download the binary under the terms of the Neural Magic Community License.


## Quick Tour and Documentation
[TODO ENGINEERING: EDIT AS NEEDED]

Follow the quick tour below to get started.
For a more in-depth read, check out [Engine documentation](https://docs.neuralmagic.com/engine/).

<!--- the docs url will become active once Marketing configures it. --->

### Installation and Requirements

[THE FOLLOWING TEXT IS IS PROCESS AS CE PROCESS IS BEING CONFIRMED]
- Installation of [SparseZoo](https://docs.neuralmagic.com/sparsezoo/) 
- Python 3.6.0 or higher
- Use Case: Computer Vision - Image Classification, Object Detection
- Model Architectures: Deep Learning Neural Network Architectures (e.g., CNNs, DNNs - refer to SparseZoo for examples)
- Instruction Set: Ideally CPUs with AVX-512 (e.g., Intel Xeon Cascade Lake, Icelake, Skylake) and 2 FMAs. VNNI support required for sparse quantization; AVX2 instruction set will work but may show less performant results.
- OS / Environment: Ubuntu, CentOS, RHEL, Amazon Linux 

To install, these packages will be required:

```python
$ pip install engine
$ pip install sparsezoo
```
General instructions for Python installation are found [here](https://realpython.com/installing-python/).

Optionally, you may also want to install the [Neural Magic Inference Engine](https://docs.neuralmagic.com/[ENGINE_REPO_NAME]/) and [Sparsify](https://docs.neuralmagic.com/sparsify/). The Engine can utilize Neural Magic’s runtime engine and Sparsify to achieve faster inference timings. For example, after running a benchmark, you might want to change optimization values and run an optimization profile again.

```python
$ pip install engine
$ pip install sparsify
```
Additionally, it is recommended to work within a virtual environment. 
Sample commands for creating and activating in a Unix-based system are provided below:
```
pip3 install virtualenv
python3 -m venv ./venv
source ./venv/bin/activate
```
1. Navigate to the parent directory of the `engine` codebase.
2. Use pip install to run the setup.py file in the repo: `pip install engine-python/`
3. Import engine library in your code: `import engine`

Note: If you run into issues with TensorFlow/PyTorch imports (specifically GPU vs. CPU support), 
you can edit the `requirements.txt` file at the root of the repository for the desired TensorFlow or PyTorch version.

## Usage

[IN PROCESS; NEEDS ENG INPUT]

## Tutorials

[IN PROCESS; NEEDS ENG INPUT]

  
## Available Models and Recipes
If you are not ready to upload your model through Engine, a number of pre-trained models in the [SparseZoo](https://docs.neuralmagic.com/sparsezoo/) can be used. Included are both baseline and recalibrated models for higher performance. These can optionally be used with [Neural Magic Inference Engine](https://github.com/neuralmagic/engine/). The types available for each model architecture are noted in the [SparseZoo model repository listing](https://docs.neuralmagic.com/sparsezoo/available-models).


## Resources and Learning More
* [Engine Documentation](https://docs.neuralmagic.com/engine/)
* [SparseZoo Documentation](https://docs.neuralmagic.com/sparsezoo/)
* [Neural Magic Blog](https://www.neuralmagic.com/blog/)
* [Neural Magic](https://www.neuralmagic.com/)

[TODO ENGINEERING: table with links for deeper topics or other links that should be included above]

## Contributing

We appreciate contributions to the documentation!

- Report issues and bugs directly in [this GitHub project](https://github.com/neuralmagic/engine/issues).

Give Engine a shout out on social! Are you able write a blog post, do a lunch ’n learn, host a meetup, or simply share via your networks? Help us build the community, yay! Here’s some details to assist:
- item 1 [TODO MARKETING: NEED METHODS]
- item n

## Join the Community

For user help or questions about Engine, please use our [GitHub Discussions](https://www.github.com/neuralmagic/engine/issues). Everyone is welcome!

You can get the latest news, webinar invites, and other ML Performance tidbits by [connecting with the Neural Magic community](https://www.neuralmagic.com/NEED_URL/).[TODO MARKETING: NEED METHOD]

For more general questions about Neural Magic please contact us this way [Method](URL). [TODO MARKETING: NEED METHOD]

[TODO MARKETING: Example screenshot here]

## License

The project is licensed under the [Neural Magic Community License 1.0](LICENSE-NEURALMAGIC).

## Release History

[Track this project via GitHub Releases.](https://github.com/neuralmagic/engine/releases)