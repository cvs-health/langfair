<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/develop/assets/images/langfair-logo.png" />
</p>

# Library for Assessing Bias and Fairness in LLMs

LangFair is a Python library for conducting bias and fairness assessments of LLM use cases. This repository includes a framework for [choosing bias and fairness metrics](https://github.com/cvs-health/langfair/tree/develop#choosing-bias-and-fairness-metrics-for-an-llm-use-case), [demo notebooks](https://github.com/cvs-health/langfair/tree/develop/examples), and a LLM bias and fairness [technical playbook](https://arxiv.org/pdf/2407.10853) containing a thorough discussion of LLM bias and fairness risks, evaluation metrics, and best practices. Please refer to our [documentation site](https://cvs-health.github.io/langfair/) for more details on how to use LangFair.

Bias and fairness metrics offered by LangFair fall into one of several categories. The full suite of metrics is displayed below.

##### Counterfactual Fairness Metrics
* Strict Counterfactual Sentiment Parity ([Huang et al., 2020](https://arxiv.org/pdf/1911.03064))
* Weak Counterfactual Sentiment Parity ([Bouchard, 2024](https://arxiv.org/pdf/2407.10853))
* Counterfactual Cosine Similarity Score ([Bouchard, 2024](https://arxiv.org/pdf/2407.10853))
* Counterfactual BLEU ([Bouchard, 2024](https://arxiv.org/pdf/2407.10853))
* Counterfactual ROUGE-L ([Bouchard, 2024](https://arxiv.org/pdf/2407.10853))

##### Stereotype Metrics
* Stereotypical Associations ([Liang et al., 2023](https://arxiv.org/pdf/2211.09110))
* Co-occurrence Bias Score ([Bordia & Bowman, 2019](https://aclanthology.org/N19-3002.pdf))
* Stereotype classifier metrics ([Zekun et al., 2023](https://arxiv.org/ftp/arxiv/papers/2311/2311.14126.pdf), [Bouchard, 2024](https://arxiv.org/pdf/2407.10853))

##### Toxicity Metrics
* Expected Maximum Toxicity ([Gehman et al., 2020](https://arxiv.org/pdf/2009.11462))
* Toxicity Probability ([Gehman et al., 2020](https://arxiv.org/pdf/2009.11462))
* Toxic Fraction ([Liang et al., 2023](https://arxiv.org/pdf/2211.09110))

##### Recommendation Fairness Metrics
* Jaccard Similarity ([Zhang et al., 2023](https://arxiv.org/pdf/2305.07609.pdf))
* Search Result Page Misinformation Score ([Zhang et al., 2023](https://arxiv.org/pdf/2305.07609.pdf))
* Pairwise Ranking Accuracy Gap ([Zhang et al., 2023](https://arxiv.org/pdf/2305.07609.pdf))

##### Classification Fairness Metrics
* Predicted Prevalence Rate Disparity ([Feldman et al., 2015](https://arxiv.org/abs/1412.3756); [Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Negative Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Omission Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Positive Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Discovery Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))

## Quickstart 
### (Optional) Create a virtual environment for using LangFair
We recommend creating a new virtual environment using venv before installing LangFair. To do so, please follow instructions [here](https://docs.python.org/3/library/venv.html).

### Installing LangFair
The latest version can be installed from PyPI:

```bash
pip install langfair
```

### Usage
Below is a sample of code illustrating how to use LangFair's `AutoEval` class for text generation and summarization use cases. The below example assumes the user has already defined parameters `DEPLOYMENT_NAME`, `API_KEY`, `API_BASE`, `API_TYPE`, `API_VERSION`, and a list of prompts from their use case `prompts`.

Create `langchain` LLM object.
```python
from langchain_openai import AzureChatOpenAI
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
    # toxicity_device=device # use if GPU is available
)

results = await auto_object.evaluate()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/develop/assets/images/autoeval_process.png" />
</p>


Print the results and export to .txt file.
```python
auto_object.export_results(file_name="metric_values.txt")
auto_object.print_results()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/develop/assets/images/autoeval_output.png" />
</p>

## Example Notebooks
See **[Demo Notebooks](https://github.com/cvs-health/langfair/tree/develop/examples)** for notebooks illustrating how to use LangFair for various bias and fairness evaluation metrics.

## Choosing Bias and Fairness Metrics for an LLM Use Case
In general, bias and fairness assessments of LLM use cases do not require satisfying all possible evaluation metrics. Instead, practitioners should prioritize and concentrate on a relevant subset of metrics. To demystify metric choice for bias and fairness assessments of LLM use cases, we introduce a decision framework for selecting the appropriate evaluation metrics, as depicted in the diagram below. Leveraging the use case taxonomy outlined in the [technical playbook](https://arxiv.org/abs/2407.10853), we determine suitable choices of bias and fairness metrics for a given use case based on its relevant characteristics.

In cases where prompts are derived from a well-defined population, we categorize the use cases into three distinct groups: 1) text generation and summarization, 2) classification, and 3) recommendation. Within each category, we carefully choose metrics to assess potential bias and fairness risks that are pertinent to the specific characteristics of the use case. The specifics of this mapping are elaborated upon below.

When considering text generation and summarization, an important factor in determining relevant bias and fairness metrics is whether the use case upholds FTUA. This means that all inputs provided to the model do not include any mentions of protected attribute words. If FTUA cannot be achieved, we recommend evaluating bias using counterfactual discrimination and stereotype metrics. We recommend that all text generation and summarization use cases undergo toxicity evaluation, regardless of whether or not FTUA is achieved. For illustrations of how to calculate these metrics, refer to the [toxicity, stereotype, and counterfactual demo notebooks](https://github.com/cvs-health/langfair/tree/develop/examples/evaluations/text_generation) or the [AutoEval demo notebook](https://github.com/cvs-health/langfair/tree/develop/examples/evaluations/text_generation/auto_eval_demo.ipynb) for a more automated implementation.

To determine suitable bias and fairness metrics for text classification use cases, we adopt a modified version of the straightforward decision framework proposed by [Aequitas](https://arxiv.org/pdf/1811.05577.pdf). This framework can be applied to any classification use case where inputs can be attributed to a protected attribute. In such cases, the following approach is recommended: if fairness necessitates that model predictions exhibit approximately equal predicted prevalence across different groups, representation fairness metrics should be used; Otherwise, error-based fairness metrics should be used. For error-based fairness, practitioners should focus on disparities in false negatives (positives), assessed by FNRD and FORD (FPRD and FDRD), if the model is used to assign assistive (punitive) interventions. In the context of fairness, if interventions are punitive, and hence can hurt individuals, it is undesirable for a model to produce false positives disproportionately for any protected attribute group. Analogously, having a model produce false negatives disproportionately for any protected attribute group is undesirable in the case of assistive interventions. If inputs cannot be mapped to a protected attribute, meaning they are not person-level inputs and they satisfy FTUA, then we argue that no fairness assessment is needed.  For such classification fairness assessments, use case owners should refer to the [classification metrics demo notebook](https://github.com/cvs-health/langfair/tree/develop/examples/evaluations/classification/classification_metrics_demo.ipynb).

For recommendation use cases, we argue that counterfactual discrimination is a risk if FTUA cannot be satisfied, as shown by [Zhang et al., 2023](https://arxiv.org/pdf/2305.07609.pdf). For these scenarios, we recommend practitioners assess counterfactual fairness in recommendations using the recommendation fairness metrics. However, if FTUA is satisfied, then we argue no fairness assessment is needed. For illustrations of how to calculte these metrics, refer to the [recommendation metrics demo notebook](https://github.com/cvs-health/langfair/tree/develop/examples/evaluations/recommendation/recommendation_metrics_demo.ipynb).

Lastly, we classify the remaining subset of focused use cases as having minimal risks in terms of bias and fairness. These use cases include tasks such as code-writing, personal learning, and any other scenarios where the use case owner manually evaluates each generated output. As these use cases inherently involve human-in-the-loop feedback, we argue that any concerns regarding bias or fairness can be adequately addressed by the use case owner who is reviewing the outputs.


<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/develop/assets/images/use_case_framework.PNG" />
</p>


## Associated Research
A technical description of LangFair's evaluation metrics and a practitioner's guide for selecting evaluation metrics is contained in **[this paper](https://arxiv.org/pdf/2407.10853)**. Below is the bibtex entry for this paper:

@misc{bouchard2024actionableframeworkassessingbias,
      title={An Actionable Framework for Assessing Bias and Fairness in Large Language Model Use Cases}, 
      author={Dylan Bouchard},
      year={2024},
      eprint={2407.10853},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.10853}, 
}

## Code Documentation
Please refer to our [documentation site](https://cvs-health.github.io/langfair/) for more details on how to use LangFair.

## Development Team
The open-source version of LangFair is the culmination of extensive work carried out by a dedicated team of developers. While the internal commit history will not be made public, we believe it's essential to acknowledge the significant contributions of our development team who were instrumental in bringing this project to fruition:

- [Dylan Bouchard](https://github.com/dylanbouchard)
- [Mohit Singh Chauhan](https://github.com/mohitcek)
- [David Skarbrevik](https://github.com/dskarbrevik)
- [Viren Bajaj](https://github.com/virenbajaj)
- [Zeya Ahmad](https://github.com/zeya30)

## Contributing
Contributions are welcome. Please refer [here](https://github.com/cvs-health/langfair/tree/develop/CONTRIBUTING.md) for instructions on how to contribute to LangFair.