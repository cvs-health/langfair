<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/main/assets/images/langfair-logo.png" />
</p>

# LangFair: Use-Case Level LLM Bias and Fairness Assessments
[![Build Status](https://github.com/cvs-health/langfair/actions/workflows/ci.yaml/badge.svg)](https://github.com/cvs-health/langfair/actions)
[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://cvs-health.github.io/langfair/latest/index.html)
[![PyPI version](https://badge.fury.io/py/langfair.svg)](https://pypi.org/project/langfair/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![](https://img.shields.io/badge/arXiv-2407.10853-B31B1B.svg)](https://arxiv.org/abs/2407.10853)


LangFair is a comprehensive Python library designed for conducting bias and fairness assessments of large language model (LLM) use cases. This repository includes a comprehensive framework for [choosing bias and fairness metrics](https://github.com/cvs-health/langfair/tree/main#choosing-bias-and-fairness-metrics-for-an-llm-use-case), along with [demo notebooks](https://github.com/cvs-health/langfair/tree/main/examples) and a [technical playbook](https://arxiv.org/abs/2407.10853) that discusses LLM bias and fairness risks, evaluation metrics, and best practices. 

Explore our [documentation site](https://cvs-health.github.io/langfair/) for detailed instructions on using LangFair.

## 🚀 Why Choose LangFair?
Static benchmark assessments, which are typically assumed to be sufficiently representative, often fall short in capturing the risks associated with all possible use cases of LLMs. These models are increasingly used in various applications, including recommendation systems, classification, text generation, and summarization. However, evaluating these models without considering use-case-specific prompts can lead to misleading assessments of their performance, especially regarding bias and fairness risks.
 
LangFair addresses this gap by adopting a Bring Your Own Prompts (BYOP) approach, allowing users to tailor bias and fairness evaluations to their specific use cases. This ensures that the metrics computed reflect the true performance of the LLMs in real-world scenarios, where prompt-specific risks are critical. Additionally, LangFair's focus is on output-based metrics that are practical for governance audits and real-world testing, without needing access to internal model states.

## ⚡ Quickstart Guide
### (Optional) Create a virtual environment for using LangFair
We recommend creating a new virtual environment using venv before installing LangFair. To do so, please follow instructions [here](https://docs.python.org/3/library/venv.html).

### Installing LangFair
The latest version can be installed from PyPI:

```bash
pip install langfair
```

### Usage Example
Below is a sample of code illustrating how to use LangFair's `AutoEval` class for text generation and summarization use cases. The below example assumes the user has already defined parameters `DEPLOYMENT_NAME`, `API_KEY`, `API_BASE`, `API_TYPE`, `API_VERSION`, and a list of prompts from their use case `prompts`. 

Create `langchain` LLM object. Any of LangChain’s LLM classes may be used in place of `AzureChatOpenAI`.
```python
from langchain_openai import AzureChatOpenAI
# import torch # uncomment if GPU is available
# device = torch.device("cuda") # uncomment if GPU is available

llm = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    azure_endpoint=API_BASE,
    openai_api_type=API_TYPE,
    openai_api_version=API_VERSION,
    temperature=0.4 # User to set temperature
)
```

Run the `AutoEval` method for automated bias / fairness evaluation
```python
from langfair.auto import AutoEval
auto_object = AutoEval(
    prompts=prompts, 
    langchain_llm=llm
    # toxicity_device=device # uncomment if GPU is available
)
results = await auto_object.evaluate()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/main/assets/images/autoeval_process.png" />
</p>


Print the results and export to .txt file.
```python
auto_object.export_results(file_name="metric_values.txt")
auto_object.print_results()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/main/assets/images/autoeval_output.png" />
</p>

## 📚 Example Notebooks
Explore the following demo notebooks to see how to use LangFair for various bias and fairness evaluation metrics:

- [Toxicity Evaluation](https://github.com/cvs-health/langfair/blob/main/examples/evaluations/text_generation/toxicity_metrics_demo.ipynb): A notebook demonstrating toxicity metrics.
- [Counterfactual Fairness Evaluation](https://github.com/cvs-health/langfair/blob/main/examples/evaluations/text_generation/counterfactual_metrics_demo.ipynb): A notebook illustrating how to generate counterfactual datasets and compute counterfactual fairness metrics.
- [Stereotype Evaluation](https://github.com/cvs-health/langfair/blob/main/examples/evaluations/text_generation/stereotype_metrics_demo.ipynb): A notebook demonstrating stereotype metrics.
- [AutoEval for Text Generation / Summarization (Toxicity, Stereotypes, Counterfactual)](https://github.com/cvs-health/langfair/blob/main/examples/evaluations/text_generation/auto_eval_demo.ipynb): A notebook illustrating how to use LangFair's `AutoEval` class for a comprehensive fairness assessment of text generation / summarization use cases. This assessment includes toxicity, stereotype, and counterfactual metrics.
- [Classification Fairness Evaluation](https://github.com/cvs-health/langfair/blob/main/examples/evaluations/classification/classification_metrics_demo.ipynb): A notebook demonstrating classification fairness metrics.
- [Recommendation Fairness Evaluation](https://github.com/cvs-health/langfair/blob/main/examples/evaluations/recommendation/recommendation_metrics_demo.ipynb): A notebook demonstrating recommendation fairness metrics.


## 🛠 Choosing Bias and Fairness Metrics for an LLM Use Case
Selecting the appropriate bias and fairness metrics is essential for accurately assessing the performance of large language models (LLMs) in specific use cases. Instead of attempting to compute all possible metrics, practitioners should focus on a relevant subset that aligns with their specific goals and the context of their application.

Our decision framework for selecting appropriate evaluation metrics is illustrated in the diagram below. For more details, refer to our [technical playbook](https://arxiv.org/abs/2407.10853).

<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/main/assets/images/use_case_framework.PNG" />
</p>

**Note:** Fairness through unawareness means none of the prompts for an LLM use case include any mention of protected attribute words.

## 📊 Supported Bias and Fairness Metrics
Bias and fairness metrics offered by LangFair are grouped into several categories. The full suite of metrics is displayed below.

##### Toxicity Metrics
* Expected Maximum Toxicity ([Gehman et al., 2020](https://arxiv.org/abs/2009.11462))
* Toxicity Probability ([Gehman et al., 2020](https://arxiv.org/abs/2009.11462))
* Toxic Fraction ([Liang et al., 2023](https://arxiv.org/abs/2211.09110))

##### Counterfactual Fairness Metrics
* Strict Counterfactual Sentiment Parity ([Huang et al., 2020](https://arxiv.org/abs/1911.03064))
* Weak Counterfactual Sentiment Parity ([Bouchard, 2024](https://arxiv.org/abs/2407.10853))
* Counterfactual Cosine Similarity Score ([Bouchard, 2024](https://arxiv.org/abs/2407.10853))
* Counterfactual BLEU ([Bouchard, 2024](https://arxiv.org/abs/2407.10853))
* Counterfactual ROUGE-L ([Bouchard, 2024](https://arxiv.org/abs/2407.10853))

##### Stereotype Metrics
* Stereotypical Associations ([Liang et al., 2023](https://arxiv.org/abs/2211.09110))
* Co-occurrence Bias Score ([Bordia & Bowman, 2019](https://arxiv.org/abs/1904.03035))
* Stereotype classifier metrics ([Zekun et al., 2023](https://arxiv.org/abs/2311.14126), [Bouchard, 2024](https://arxiv.org/abs/2407.10853))

##### Recommendation (Counterfactual) Fairness Metrics
* Jaccard Similarity ([Zhang et al., 2023](https://dl.acm.org/doi/10.1145/3604915.3608860))
* Search Result Page Misinformation Score ([Zhang et al., 2023](https://dl.acm.org/doi/10.1145/3604915.3608860))
* Pairwise Ranking Accuracy Gap ([Zhang et al., 2023](https://dl.acm.org/doi/10.1145/3604915.3608860))

##### Classification Fairness Metrics
* Predicted Prevalence Rate Disparity ([Feldman et al., 2015](https://arxiv.org/abs/1412.3756); [Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Negative Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Omission Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Positive Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Discovery Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))


## 📖 Associated Research
A technical description of LangFair's evaluation metrics and a practitioner's guide for selecting evaluation metrics is contained in **[this paper](https://arxiv.org/abs/2407.10853)**. If you use our framework for selecting evaluation metrics, we would appreciate citations to the following paper:

```bibtex
@misc{bouchard2024actionableframeworkassessingbias,
      title={An Actionable Framework for Assessing Bias and Fairness in Large Language Model Use Cases}, 
      author={Dylan Bouchard},
      year={2024},
      eprint={2407.10853},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.10853}, 
}
```

## 📄 Code Documentation
Please refer to our [documentation site](https://cvs-health.github.io/langfair/) for more details on how to use LangFair.

## 🤝 Development Team
The open-source version of LangFair is the culmination of extensive work carried out by a dedicated team of developers. While the internal commit history will not be made public, we believe it's essential to acknowledge the significant contributions of our development team who were instrumental in bringing this project to fruition:

- [Dylan Bouchard](https://github.com/dylanbouchard)
- [Mohit Singh Chauhan](https://github.com/mohitcek)
- [David Skarbrevik](https://github.com/dskarbrevik)
- [Viren Bajaj](https://github.com/virenbajaj)
- [Zeya Ahmad](https://github.com/zeya30)

## 🤗 Contributing
Contributions are welcome. Please refer [here](https://github.com/cvs-health/langfair/tree/main/CONTRIBUTING.md) for instructions on how to contribute to LangFair.