<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/main/assets/images/langfair-logo.png" />
</p>

# LangFair: Use-Case Level LLM Bias and Fairness Assessments
[![Build Status](https://github.com/cvs-health/langfair/actions/workflows/ci.yaml/badge.svg)](https://github.com/cvs-health/langfair/actions)
[![PyPI version](https://badge.fury.io/py/langfair.svg)](https://pypi.org/project/langfair/)
[![Downloads](https://img.shields.io/pepy/dt/langfair)](https://pepy.tech/projects/langfair?timeRange=threeMonths&category=version&includeCIDownloads=true&granularity=daily&viewType=line&versions=0.6.3%2C0.6.2%2C0.6.1)
[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://cvs-health.github.io/langfair/latest/index.html)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![](https://img.shields.io/badge/arXiv-2407.10853-B31B1B.svg)](https://arxiv.org/abs/2407.10853)


LangFair is a comprehensive Python library designed for conducting bias and fairness assessments of large language model (LLM) use cases. This repository includes various supporting resources, including

- [Documentation site](https://cvs-health.github.io/langfair/) with complete API reference
- [Comprehensive framework](https://github.com/cvs-health/langfair/tree/main#-choosing-bias-and-fairness-metrics-for-an-llm-use-case) for choosing bias and fairness metrics
- [Demo notebooks](https://github.com/cvs-health/langfair/tree/main#-example-notebooks) providing illustrative examples
- [LangFair tutorial](https://medium.com/cvs-health-tech-blog/how-to-assess-your-llm-use-case-for-bias-and-fairness-with-langfair-7be89c0c4fab) on Medium
- [Software paper](https://arxiv.org/abs/2501.03112v1) on how LangFair compares to other toolkits
- [Research paper](https://arxiv.org/abs/2407.10853) on our evaluation approach

## 🚀 Why Choose LangFair?
Static benchmark assessments, which are typically assumed to be sufficiently representative, often fall short in capturing the risks associated with all possible use cases of LLMs. These models are increasingly used in various applications, including recommendation systems, classification, text generation, and summarization. However, evaluating these models without considering use-case-specific prompts can lead to misleading assessments of their performance, especially regarding bias and fairness risks.
 
LangFair addresses this gap by adopting a Bring Your Own Prompts (BYOP) approach, allowing users to tailor bias and fairness evaluations to their specific use cases. This ensures that the metrics computed reflect the true performance of the LLMs in real-world scenarios, where prompt-specific risks are critical. Additionally, LangFair's focus is on output-based metrics that are practical for governance audits and real-world testing, without needing access to internal model states.

<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/release-branch/v0.4.0/assets/images/langfair_graphic.png" />
</p>

**Note:** This diagram illustrates the workflow for assessing bias and fairness in text generation and summarization use cases.

## ⚡ Quickstart Guide
### (Optional) Create a virtual environment for using LangFair
We recommend creating a new virtual environment using venv before installing LangFair. To do so, please follow instructions [here](https://docs.python.org/3/library/venv.html).

### Installing LangFair
The latest version can be installed from PyPI:

```bash
pip install langfair
```

### Usage Examples
Below are code samples illustrating how to use LangFair to assess bias and fairness risks in text generation and summarization use cases. The below examples assume the user has already defined a list of prompts from their use case, `prompts`. 

##### Generate LLM responses
To generate responses, we can use LangFair's `ResponseGenerator` class. First, we must create a `langchain` LLM object. Below we use `ChatVertexAI`, but **any of [LangChain’s LLM classes](https://js.langchain.com/docs/integrations/chat/) may be used instead**. Note that `InMemoryRateLimiter` is to used to avoid rate limit errors.
```python
from langchain_google_vertexai import ChatVertexAI
from langchain_core.rate_limiters import InMemoryRateLimiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4.5, check_every_n_seconds=0.5, max_bucket_size=280,  
)
llm = ChatVertexAI(
    model_name="gemini-pro", temperature=0.3, rate_limiter=rate_limiter
)
```
We can use `ResponseGenerator.generate_responses` to generate 25 responses for each prompt, as is convention for toxicity evaluation.
```python
from langfair.generator import ResponseGenerator
rg = ResponseGenerator(langchain_llm=llm)
generations = await rg.generate_responses(prompts=prompts, count=25)
responses = generations["data"]["response"]
duplicated_prompts = generations["data"]["prompt"] # so prompts correspond to responses
```

##### Compute toxicity metrics
Toxicity metrics can be computed with `ToxicityMetrics`. Note that use of `torch.device` is optional and should be used if GPU is available to speed up toxicity computation.
```python
# import torch # uncomment if GPU is available
# device = torch.device("cuda") # uncomment if GPU is available
from langfair.metrics.toxicity import ToxicityMetrics
tm = ToxicityMetrics(
    # device=device, # uncomment if GPU is available,
)
tox_result = tm.evaluate(
    prompts=duplicated_prompts, 
    responses=responses, 
    return_data=True
)
tox_result['metrics']
# # Output is below
# {'Toxic Fraction': 0.0004,
# 'Expected Maximum Toxicity': 0.013845130120171235,
# 'Toxicity Probability': 0.01}
```

##### Compute stereotype metrics
Stereotype metrics can be computed with `StereotypeMetrics`.
```python
from langfair.metrics.stereotype import StereotypeMetrics
sm = StereotypeMetrics()
stereo_result = sm.evaluate(responses=responses, categories=["gender"])
stereo_result['metrics']
# # Output is below
# {'Stereotype Association': 0.3172750176745329,
# 'Cooccurrence Bias': 0.44766333654278373,
# 'Stereotype Fraction - gender': 0.08}
```

##### Generate counterfactual responses and compute metrics
We can generate counterfactual responses with `CounterfactualGenerator`.
```python
from langfair.generator.counterfactual import CounterfactualGenerator
cg = CounterfactualGenerator(langchain_llm=llm)
cf_generations = await cg.generate_responses(
    prompts=prompts, attribute='gender', count=25
)
male_responses = cf_generations['data']['male_response']
female_responses = cf_generations['data']['female_response']
```

Counterfactual metrics can be easily computed with `CounterfactualMetrics`.
```python
from langfair.metrics.counterfactual import CounterfactualMetrics
cm = CounterfactualMetrics()
cf_result = cm.evaluate(
    texts1=male_responses, 
    texts2=female_responses,
    attribute='gender'
)
cf_result['metrics']
# # Output is below
# {'Cosine Similarity': 0.8318708,
# 'RougeL Similarity': 0.5195852482361165,
# 'Bleu Similarity': 0.3278433712872481,
# 'Sentiment Bias': 0.0009947145187601957}
```

##### Alternative approach: Semi-automated evaluation with `AutoEval`
To streamline assessments for text generation and summarization use cases, the `AutoEval` class conducts a multi-step process that completes all of the aforementioned steps with two lines of code.
```python
from langfair.auto import AutoEval
auto_object = AutoEval(
    prompts=prompts, 
    langchain_llm=llm,
    # toxicity_device=device # uncomment if GPU is available
)
results = await auto_object.evaluate()
results['metrics']
# # Output is below
# {'Toxicity': {'Toxic Fraction': 0.0004,
#   'Expected Maximum Toxicity': 0.013845130120171235,
#   'Toxicity Probability': 0.01},
#  'Stereotype': {'Stereotype Association': 0.3172750176745329,
#   'Cooccurrence Bias': 0.44766333654278373,
#   'Stereotype Fraction - gender': 0.08,
#   'Expected Maximum Stereotype - gender': 0.60355167388916,
#   'Stereotype Probability - gender': 0.27036},
#  'Counterfactual': {'male-female': {'Cosine Similarity': 0.8318708,
#    'RougeL Similarity': 0.5195852482361165,
#    'Bleu Similarity': 0.3278433712872481,
#    'Sentiment Bias': 0.0009947145187601957}}}
```

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

Our decision framework for selecting appropriate evaluation metrics is illustrated in the diagram below. For more details, refer to our [research paper](https://arxiv.org/abs/2407.10853) detailing the evaluation approach.

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
A technical description and a practitioner's guide for selecting evaluation metrics is contained in **[this paper](https://arxiv.org/abs/2407.10853)**. If you use our evaluation approach, we would appreciate citations to the following paper:

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

A high-level description of LangFair's functionality is contained in **[this paper](https://arxiv.org/abs/2501.03112)**. If you use LangFair, we would appreciate citations to the following paper:

```bibtex
@misc{bouchard2025langfairpythonpackageassessing,
      title={LangFair: A Python Package for Assessing Bias and Fairness in Large Language Model Use Cases}, 
      author={Dylan Bouchard and Mohit Singh Chauhan and David Skarbrevik and Viren Bajaj and Zeya Ahmad},
      year={2025},
      eprint={2501.03112},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.03112}, 
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