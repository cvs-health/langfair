"""
.. _auto_eval_demo:

===============================================================
Auto Eval Demo - Dialogue Summarization
===============================================================

"""
# %%
# This notebook demonstrate the implementation of ``AutoEval`` class. This class provides an user-friendly way to compute toxicity, stereotype, and counterfactual assessment for an LLM model. The user needs to provide the input prompts and model responses (optional) and the ``AutoEval`` class implement following steps.
#
# 1. Check Fairness Through Awareness (FTU)
# 2. If FTU is not satisfied, generate dataset for Counterfactual assessment
# 3. If not provided, generate model responses
# 4. Compute toxicity metrics
# 5. Compute stereotype metrics
# 6. If FTU is not satisfied, compute counterfactual metrics
#
# Import necessary python libraries, suppress benign warnings, and specify the model API key.


# Run if python-dotenv not installed
# import sys
# !{sys.executable} -m pip install python-dotenv

import os
import warnings

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter

from langfair.auto import AutoEval

warnings.filterwarnings("ignore")


# User to populate .env file with API credentials
repo_path = "/".join(os.getcwd().split("/")[:-3])
load_dotenv(find_dotenv())

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
API_TYPE = os.getenv("API_TYPE")
API_VERSION = os.getenv("API_VERSION")
MODEL_VERSION = os.getenv("MODEL_VERSION")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# %%
# Here we read in a sample of conversation/dialogue between a person and a doctor from the
# `Neil Code Dialogsum-test <https://32a20588.isolation.zscaler.com/profile/a0ca9a0d-8973-4cbe-8155-e152179e8291/zia-session/?controls_id=0731d209-a26f-4f9a-9cb0-4fdc914a6ee6&region=was&tenant=2d433b801dec&user=f14ec5bc375d9c4122780b06db815ffcacff56adb229b59b6a459dd1718e0c91&original_url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fneil-code%2Fdialogsum-test%2Fblob%2Fmain%2FREADME.md&key=sh-1&hmac=0abf7b681024a518be4227d7bee5186dfb34c133fbd0922d1795f0394a48b818>`_.
# Update the following cell to read input prompts and (if applicable) model responses as python list.


benchmark_path = os.path.join(repo_path, "data/neil_code_dialogsum_train.txt")

with open(benchmark_path, "r") as file:
    dialogue = []
    for line in file:
        dialogue.append(line)

print("Number of Dialogues: ", len(dialogue))
dialogue[:5]

INSTRUCTION = (
    "You are to summarize the following conversation in no more than 3 sentences: \n"
)
prompts = [INSTRUCTION + str(text) for text in dialogue[:100]]

# %%
# ``AutoEval()`` - For calculating all toxicity, stereotype, and counterfactual metrics supported by LangFair
#
# **Class Attributes:**
#
# * ``prompts`` - (**list of strings**) A list of input prompts for the model.
#
# * ``responses`` - (**list of strings, default=None**) A list of generated output from an LLM. If not available, responses are computed using the model.
#
# * ``langchain_llm`` (**langchain llm (Runnable), default=None**) A langchain llm object to get passed to LLMChain `llm` argument.
#
# * ``suppressed_exceptions``` (**tuple, default=None**) Specifies which exceptions to handle as 'Unable to get response' rather than raising the exception
#
# * ``metrics`` - (**dict or list of str, default is all metrics**) Specifies which metrics to evaluate.
#
# * ``toxicity_device`` - (**str or torch.device input or torch.device object, default="cpu"**) Specifies the device that toxicity classifiers use for prediction. Set to "cuda" for classifiers to be able to leverage the GPU. Currently, 'detoxify_unbiased' and 'detoxify_original' will use this parameter.
#
# * ``neutralize_tokens`` - (**bool, default=True**) An indicator attribute to use masking for the computation of Blue and RougeL metrics. If True, counterfactual responses are masked using `CounterfactualGenerator.neutralize_tokens` method before computing the aforementioned metrics.
#
# * ``max_calls_per_min`` (**Deprecated as of 0.2.0**) Use LangChain's InMemoryRateLimiter instead.
#
# **Class Methods:**
#
# 1. ``evaluate`` - Compute supported metrics.
#
#     **Method Attributes:**
#     - ``metrics`` - (**dict or list of str, default=None**)
#     Specifies which metrics to evaluate if a change is desired from those specified in self.metrics.
#
# 2. ``print_results`` - Print evaluated score in a clean format.
#
# 3. ``export_results`` - Save the final result in a text file.
#
#     **Method Attributes:**
#     - ``file_name`` - (**str, default="results.txt"**)
#     Name of the .txt file.
#
# Below we use LangFair's ``AutoEval`` class to conduct a comprehensive bias and fairness assessment for our text generation/summarization use case. To instantiate the `AutoEval` class, provide prompts and LangChain LLM object.
#
# **Important note: We provide three examples of LangChain LLMs below, but these can be replaced with a LangChain LLM of your choice.**

# Use LangChain's InMemoryRateLimiter to avoid rate limit errors. Adjust parameters as necessary.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=10,
    check_every_n_seconds=10,
    max_bucket_size=1000,
)

# %%
# **Example 1: Gemini Pro with VertexAI**

# # Run if langchain-google-vertexai not installed. Note: kernel restart may be required.
# import sys
# !{sys.executable} -m pip install langchain-google-vertexai

# from langchain_google_vertexai import VertexAI
# llm = VertexAI(model_name='gemini-pro', temperature=1, rate_limiter=rate_limiter)

# # Define exceptions to suppress
# suppressed_exceptions = (IndexError, ) # suppresses error when gemini refuses to answer

# %%
# **Example 2: Mistral AI**

# # Run if langchain-mistralai not installed. Note: kernel restart may be required.
# import sys
# !{sys.executable} -m pip install langchain-mistralai

# os.environ["MISTRAL_API_KEY"] = os.getenv('M_KEY')
# from langchain_mistralai import ChatMistralAI

# llm = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=1,
#     rate_limiter=rate_limiter
# )
# suppressed_exceptions = None

# %%
# **Example 3: OpenAI on Azure**


# # Run if langchain-openai not installed
# import sys
# !{sys.executable} -m pip install langchain-openai

import openai
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    azure_endpoint=API_BASE,
    openai_api_type=API_TYPE,
    openai_api_version=API_VERSION,
    temperature=1,  # User to set temperature
    rate_limiter=rate_limiter,
)

# Define exceptions to suppress
suppressed_exceptions = (
    openai.BadRequestError,
    ValueError,
)  # this suppresses content filtering errors

# %%
# Instantiate ``AutoEval`` class


# import torch # uncomment if GPU is available
# device = torch.device("cuda") # uncomment if GPU is available
ae = AutoEval(
    prompts=prompts,  # small sample used as an example; in practice, a bigger sample should be used
    langchain_llm=llm,
    suppressed_exceptions=suppressed_exceptions,
    # toxicity_device=device # uncomment if GPU is available
)

# %%
# Call ``evaluate`` method to compute scores corresponding to supported metrics.
#
# Note that this  may take some time due to evaluation being computationally intensive. Consider using GPU acceleration for  faster processing.


warnings.filterwarnings("ignore")
results = await ae.evaluate()

# %%
# The ``evaluate`` method return the score computed for different metrics and also store as an attribute of ``AutoEval`` class object (``AutoEval.results``). The ``results`` attribute can be printed in a clean format using ``print_results`` method.


ae.print_results()

# %%
# The ``export_results`` method stores the above method results in a .txt file.


ae.export_results(file_name="final_metrics.txt")

# %%
# The ``AutoEval`` class also stores the toxicity and stereotype scores in a Dataframe. The dataframe can be accessed using ``toxicity_data`` and ``steretype_data`` attributes. This is useful for investigating potentially toxic or stereotypical outputs.

toxicity_data = pd.DataFrame(ae.toxicity_data)
toxicity_data.sort_values(by="score", ascending=False).head()


stereotype_data = pd.DataFrame(ae.stereotype_data)
stereotype_data.sort_values(by="stereotype_score_gender", ascending=False).head()

# %%
# **Metric Definitions**
# **********************
#
# Below are details of the LLM bias / fairness evaluation metrics calculated by the ``AutoEval`` class. Metrics are defined in the context of a sample of :math:`N` LLM outputs, denoted :math:`\hat{Y}_1,...,\hat{Y}_N`.
# **Below, a  ❗ is used to indicate the metrics we deem to be of particular importance.**
#
# **Toxicity Metrics**
#
# Toxicity metrics leverage a pre-trained toxicity classifier :math:`T: \mathcal{Y} \xrightarrow[]{} [0,1]` that maps texts to toxicity scores. Below, let :math:`\tau` denote the threshold for mapping toxicity scores to :math:`\{0,1\}`. **For all toxicity metrics, values closer to 0 indicate greater fairness.**
#
# Toxic Fraction (TF) ❗
#
# Toxic fraction measures the proportion of generations that are classified as toxic.
#
# .. math ::
#       TF = \frac{1}{N} \sum_{i=1}^N I(T(\hat{Y}_i) > \tau)
#
# Responses classified as toxic can be investigated with ``AutoEval.toxicity_data()``.
#
# Expected Maximum Toxicity (EMT)
#
# EMT estimates the maximum predicted toxicity probability among the top 25 generations:
#
# .. math ::
#       EMT = \frac{1}{N} \sum_{i=1}^N  \max_{ 1 \leq j \leq 25} T(\hat{Y}_{ij})
#
# Toxicity Probability (TP)
#
# TP is calculated as the empirical probability of having at least one toxic prediction among the top 25 generations:
#
# .. math ::
#       TP = \frac{1}{N} \sum_{i=1}^N I( \max_{ 1 \leq j \leq 25} T (\hat{Y}_{ij}) \geq \tau)
#
# **Stereotype Metrics**
#
# Stereotype metrics either leverage a pre-trained stereotype classifier :math:`St: \mathcal{Y} \xrightarrow[]{} [0,1]` that maps texts to stereotype scores **or** calculate stereotype likelihood based on word co-occurrences. Below, let :math:`\tau` denote the threshold for mapping stereotype scores to :math:`\{0,1\}`. **For all stereotype metrics, values closer to 0 indicate greater fairness.**
#
# Stereotype Fraction (SF)  ❗
#
# Stereotype fraction measures the proportion of generations that are classified as stereotypes.
#
# .. math ::
#      SF = \frac{1}{N} \sum_{i=1}^N I(St(\hat{Y}_i) > \tau)
#
# Expected Maximum Stereotype (EMS)
#
# EMS estimates the maximum predicted toxicity probability among the top 25 generations:
#
# .. math ::
#       EMS = \frac{1}{N} \sum_{i=1}^N  \max_{ 1 \leq j \leq 25} T(\hat{Y}_{ij})
#
# Responses classified as stereotypes can be investigated with ``AutoEval.stereotype_data()``.
#
# Stereotype Probability (SP)
#
# SP is calculated as the empirical probability of having at least one stereotype among the top 25 generations:
#
# .. math ::
#       SP = \frac{1}{N} \sum_{i=1}^N I( \max_{ 1 \leq j \leq 25} St (\hat{Y}_{ij}) \geq \tau)
#
# Cooccurrence Bias Score (COBS)
#
# Given two protected attribute groups :math:`G', G''` with associated sets of protected attribute words :math:`A', A''`, a set of stereotypical words :math:`W`, COBS computes the relative likelihood that an LLM :math:`\mathcal{M}` generates output having co-occurrence of :math:`w \in W` with :math:`A'` versus :math:`A''`:
#
# .. math ::
#       COBS = \frac{1}{|W|} \sum_{w \in W} \log \frac{P(w|A')}{P(w|A'')}
#
# Stereotypical Associations (SA)
#
# Consider a set of protected attribute groups :math:`\mathcal{G}`, an associated set of protected attribute lexicons :math:`\mathcal{A}`, and an associated set of stereotypical words :math:`W`. Additionally, let :math:`C(x,\hat{Y})` denote the number of times that the word :math:`x` appears in the output :math:`\hat{Y}, I(\cdot)` denote the indicator function, :math:`P^{\text{ref}}` denote a reference distribution, and :math:`TVD` denote total variation difference. SA measures the relative co-occurrence of a set of stereotypically associated words across protected attribute groups:
#
# .. math ::
#       SA = \frac{1}{|W|}\sum_{w \in W} TVD(P^{(w)},P^{\text{ref}}).
#
# where
#
# .. math ::
#       P^{\text{ref}} = \{ \frac{\sum_{A \in \mathcal{A}} \gamma(w | A)}{\sum_{A \in \mathcal{A}} \sum_{w \in W} \gamma(w | A)} : w \in W \}, \quad \gamma{(w | A)} = \sum_{a \in A} \sum_{i=1}^N C(a,\hat{Y}_i)I(C(w,\hat{Y}_i)>0).
#
#
# Counterfactual Fairness Metrics
#
# Given two protected attribute groups :math:`G', G''`, a counterfactual input pair is defined as a pair of prompts, :math:`X_i', X_i''` that are identical in every way except the former mentions protected attribute group :math:`G'` and the latter mentions :math:`G''`. Counterfactual metrics are evaluated on a sample of counterfactual response pairs :math:`(\hat{Y}_1', \hat{Y}_1''),...,(\hat{Y}_N', \hat{Y}_N'')` generated by an LLM from a sample of counterfactual input pairs :math:`(X_1',X_1''),...,(X_N',X_N'')`.
#
# *Counterfactual Similarity Metrics*
#
# Counterfactual similarity metrics assess similarity of counterfactually generated outputs. For the below three metrics, **values closer to 1 indicate greater fairness.**
#
# Counterfactual ROUGE-L (CROUGE-L)  ❗
#
# CROUGE-L is defined as the average ROUGE-L score over counterfactually generated output pairs:
#
# .. math ::
#       CROUGE-L =  \frac{1}{N} \sum_{i=1}^N \frac{2r_i'r_i''}{r_i' + r_i''},
# where
#
# .. math ::
#       r_i' = \frac{LCS(\hat{Y}_i', \hat{Y}_i'')}{len (\hat{Y}_i') }, \quad r_i'' = \frac{LCS(\hat{Y}_i'', \hat{Y}_i') }{len (\hat{Y}_i'') }
#
# where :math:`LCS(\cdot,\cdot)` denotes the longest common subsequence of tokens between two LLM outputs, and :math:`len (\hat{Y})` denotes the number of tokens in an LLM output. The CROUGE-L metric effectively uses ROUGE-L to assess similarity as the longest common subsequence (LCS) relative to generated text length. For more on interpreting ROUGE-L scores, refer to `Klu.ai documentation <https://klu.ai/glossary/rouge-score#:~:text=A%20good%20ROUGE%20score%20varies,low%20at%200.3%20to%200.4.>`_.
#
# Counterfactual BLEU (CBLEU) ❗
#
# CBELEU is defined as the average BLEU score over counterfactually generated output pairs:
#
# .. math ::
#       CBLEU =  \frac{1}{N} \sum_{i=1}^N \min(BLEU(\hat{Y}_i', \hat{Y}_i''), BLEU(\hat{Y}_i'', \hat{Y}_i')).
#
# For more on interpreting BLEU scores, refer to `Google's documentation <https://cloud.google.com/translate/automl/docs/evaluate>`_.
#
# Counterfactual Cosine Similarity (CCS)  ❗
#
# Given a sentence transformer :math:`\mathbf{V} : \mathcal{Y} \xrightarrow{} \mathbb{R}^d`, CCS is defined as the average cosine simirity score over counterfactually generated output pairs:
#
# .. math ::
#       CCS = \frac{1}{N} \sum_{i=1}^N   \frac{\mathbf{V}(Y_i') \cdot \mathbf{V}(Y_i'') }{ \lVert \mathbf{V}(Y_i') \rVert \lVert \mathbf{V}(Y_i'') \rVert},
#
# *Counterfactual Sentiment Metrics*
#
# Counterfactual sentiment metrics leverage a pre-trained sentiment classifier :math:`Sm: \mathcal{Y} \xrightarrow[]{} [0,1]` to assess sentiment disparities of counterfactually generated outputs. For the below three metrics, **values closer to 0 indicate greater fairness.**
#
# Counterfactual Sentiment Bias (CSB)  ❗
#
# CSP calculates Wasserstein-1 distance \citep{wasserstein} between the output distributions of a sentiment classifier applied to counterfactually generated LLM outputs:
#
# .. math ::
#       CSB = \mathbb{E}_{\tau \sim \mathcal{U}(0,1)} | Sm(\hat{Y}') - Sm(\hat{Y}'')|,
#
# where :math:`\mathcal{U}(0,1)` denotes the uniform distribution. Above, :math:`\mathbb{E}_{\tau \sim \mathcal{U}(0,1)}` is calculated empirically on a sample of counterfactual response pairs :math:`(\hat{Y}_1', \hat{Y}_1''),...,(\hat{Y}_N', \hat{Y}_N'')` generated by :math:`\mathcal{M}`, from a sample of counterfactual input pairs :math:`(X_1',X_1''),...,(X_N',X_N'')` drawn from :math:`\mathcal{P}_{X|\mathcal{A}}`.
