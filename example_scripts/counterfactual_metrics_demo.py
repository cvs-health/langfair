"""
.. _counterfactual_metrics:

===============================================================
Counterfactual Metrics
===============================================================

"""

# %%
# .. warning ::
#   Due to the topic of bias and fairness, some users may be offended by the content contained herein, including prompts and output generated from use of the prompts.
#
# Counterfactual Assessment Metrics
# ---------------------------------
#
# Content
# *******
#
# 1. :ref:`Introduction<intro>`
#
# 2. :ref:`Generate Demo Dataset<gen-demo-dataset>`
#
# 3. :ref:`Assessment<assessment>`
#
#    * 3.1 :ref:`Lazy Implementation<lazy>`
#
#    * 3.2 :ref:`Separate Implementation<separate>`
#
# 4. :ref:`Metric Definitions<metric-defns>`
#
# Import necessary libraries for the notebook.

# Run if python-dotenv not installed
# import sys
# !{sys.executable} -m pip install python-dotenv

import json
import os
from itertools import combinations

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter

from langfair.generator.counterfactual import CounterfactualGenerator
from langfair.metrics.counterfactual import CounterfactualMetrics
from langfair.metrics.counterfactual.metrics import (
    BleuSimilarity,
    CosineSimilarity,
    RougelSimilarity,
    SentimentBias,
)

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
# .. _intro:
# 1. Introduction
# ----------------
# In many contexts, it is undesirable for a large language model (LLM) to generate substantially different output as a result of different protected attribute words contained in the input prompts, all else equal. This concept is known as (lack of) counterfactual fairness. LangFair offers the following counterfactual fairness metrics from the LLM fairness literature:
#
# * Strict Counterfactual Sentiment Parity `Huang et al., 2020 <https://arxiv.org/pdf/1911.03064>`_
# * Weak Counterfactual Sentiment Parity `Bouchard, 2024 <https://arxiv.org/pdf/2407.10853>`_
# * Counterfactual Cosine Similarity Score `Bouchard, 2024 <https://arxiv.org/pdf/2407.10853>`_
# * Counterfactual BLEU `Bouchard, 2024 <https://arxiv.org/pdf/2407.10853>`_
# * Counterfactual ROUGE-L `Bouchard, 2024 <https://arxiv.org/pdf/2407.10853>`_
#
# For more details on the definitions of these metrics, refer to the Metric Definitions in this notebook or LangFair's `technical playbook <https://arxiv.org/pdf/2407.10853>`_
#
# .. _gen-demo-dataset:
# 2. Generate Demo Dataset
# ----------------
#
# Load input prompts with `'race`' as sensitive attribute.

# THIS IS AN EXAMPLE SET OF PROMPTS. USER TO REPLACE WITH THEIR OWN PROMPTS
resource_path = os.path.join(repo_path, "data/RealToxicityPrompts.jsonl")
with open(resource_path, "r") as file:
    # Read each line in the file
    challenging = []
    prompts = []
    for line in file:
        # Parse the JSON object from each line
        challenging.append(json.loads(line)["challenging"])
        prompts.append(json.loads(line)["prompt"]["text"])
prompts = [prompts[i] for i in range(len(prompts)) if not challenging[i]][15000:30000]

# %%
# Counterfactual Dataset Generator
# --------------------------------
# ``CounterfactualGenerator()`` - Class for generating data for counterfactual discrimination assessment (class)
#
# **Class Attributes:**
#
# - ``langchain_llm`` (**langchain llm (Runnable), default=None**) A langchain llm object to get passed to LLMChain `llm` argument.
# - ``suppressed_exceptions`` (**tuple, default=None**) Specifies which exceptions to handle as 'Unable to get response' rather than raising the exception
# - ``max_calls_per_min`` (**deprecated as of 0.2.0**) Use LangChain's InMemoryRateLimiter instead.
#
# **Methods:**
#
# 1. ``parse_texts()`` - Parses a list of texts for protected attribute words and names
#
#     **Method Parameters:**
#
#     - ``text`` - (**string**) A text corpus to be parsed for protected attribute words and names
#     - ``attribute`` - (**{'race','gender','name'}**) Specifies what to parse for among race words, gender words, and names
#     - ``custom_list`` - (**List[str], default=None**) Custom list of tokens to use for parsing prompts. Must be provided if attribute is None.
#
#     **Returns:**
#     - list of results containing protected attribute words found (**list**)
#
# 2. ``create_prompts()`` - Creates counterfactual prompts by counterfactual substitution
#
#     **Method Parameters:**
#
#     - ``prompts`` - (**List of strings**) A list of prompts on which counterfactual substitution and response generation will be done
#     - ``attribute`` - (**{'gender', 'race'}, default=None**) Specifies what to parse for among race words and gender words. Must be specified if custom_list is None.
#     - ``custom_dict`` - (**Dict[str, List[str]], default=None**) A dictionary containing corresponding lists of tokens for counterfactual substitution. Keys should correspond to groups. Must be provided if attribute is None. For example: {'male': ['he', 'him', 'woman'], 'female': ['she', 'her', 'man']}
#             subset_prompts : bool, default=True
#
#     **Returns:**
#     - list of prompts on which counterfactual substitution was completed (**list**)
#
# 3. ``neutralize_tokens()`` - Neutralize gender and race words contained in a list of texts. Replaces gender words with a gender-neutral equivalent and race words with "[MASK]".
#
#     **Method Parameters:**
#
#     - ``text_list`` - (**List of strings**) A list of texts on which gender or race neutralization will occur
#     - ``attribute`` - (**{'gender', 'race'}, default='gender'**) Specifies whether to use race or gender for for neutralization
#
#     **Returns:**
#     - list of texts neutralized with respect to race or gender (**list**)
#
# 4. ``generate_responses()`` - Creates counterfactual prompts obtained by counterfactual substitution and generates responses asynchronously.
#
#     **Method Parameters:**
#
#     - ``prompts`` - (**List of strings**) A list of prompts on which counterfactual substitution and response generation will be done
#     - ``attribute`` - (**{'gender', 'race'}, default='gender'**) Specifies whether to use race or gender for counterfactual substitution
#     - ``system_prompt`` - (**str, default="You are a helpful assistant."**) Specifies system prompt for generation
#     - ``count`` - (**int, default=25**) Specifies number of responses to generate for each prompt.
#     - ``custom_dict`` - (**Dict[str, List[str]], default=None**) A dictionary containing corresponding lists of tokens for counterfactual substitution. Keys should correspond to groups. Must be provided if attribute is None. For example: {'male': ['he', 'him', 'woman'], 'female': ['she', 'her', 'man']}
#
#     **Returns:** A dictionary with two keys: `data` and `metadata`.
#     - ``data`` (**dict**) A dictionary containing the prompts and responses.
#     - ``metadata`` (**dict**) A dictionary containing metadata about the generation process, including non-completion rate, temperature, count, original prompts, and identified proctected attribute words.
#
# Below we use LangFair's ``CounterfactualGenerator`` class to check for fairness through unawareness, construct counterfactual prompts, and generate counterfactual LLM responses for computing metrics. To instantiate the ``CounterfactualGenerator`` class, pass a LangChain LLM object as an argument.
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
# Instantiate ``CounterfactualGenerator`` class


# Create langfair CounterfactualGenerator object
cdg = CounterfactualGenerator(
    langchain_llm=llm, suppressed_exceptions=suppressed_exceptions
)

# %%
# For illustration, this notebook assesses with 'race' as the protected attribute, but metrics can be evaluated for 'gender' or other custom protected attributes in the same way. First, the above mentioned `parse_texts` method is used to identify the input prompts that contain protected attribute words.
#
# Note: We recommend using atleast 1000 prompts that contain protected attribute words for better estimates. Otherwise, increase `count` attribute of `CounterfactualGenerator` class generate more responses.

# Check for fairness through unawareness
attribute = "race"
df = pd.DataFrame({"prompt": prompts})
df[attribute + "_words"] = cdg.parse_texts(texts=prompts, attribute=attribute)

# Remove input prompts that doesn't include a race word
race_prompts = df[df["race_words"].apply(lambda x: len(x) > 0)][
    ["prompt", "race_words"]
]
print(f"Race words found in {len(race_prompts)} prompts")
race_prompts.tail(5)

# %%
# Generate the model response on the input prompts using ``generate_responses`` method.

generations = await cdg.generate_responses(
    prompts=df["prompt"], attribute="race", count=1
)
output_df = pd.DataFrame(generations["data"])
output_df.head(1)

race_cols = ["white_response", "black_response", "asian_response", "hispanic_response"]

# Filter output to remove rows where any of the four counterfactual responses was refused
race_eval_df = output_df[
    ~output_df[race_cols].apply(lambda x: x == "Unable to get response").any(axis=1)
    | ~output_df[race_cols]
    .apply(lambda x: x.str.lower().str.contains("sorry"))
    .any(axis=1)
]


# %%
# .. _assessment:
# 3. Assessment
# ----------------
# This section shows two ways to evaluate countefactual metrics on a given dataset.
#
# 1. Lazy Implementation: Evalaute few or all available metrics on available dataset. This approach is useful for quick or first dry-run.
#
# 2. Separate Implemention: Evaluate each metric separately, this is useful to investage more about a particular metric.
#
# .. _lazy:
# 3.1 Lazy Implementation
#
#   ``CounterfactualMetrics()`` - Calculate all the counterfactual metrics (class)
#   **Class Attributes:**
#   - `metrics` - (**List of strings/Metric objects**) Specifies which metrics to use.
#   Default option is a list if strings (`metrics` = ["Cosine", "Rougel", "Bleu", "Sentiment Bias"]).
#   - `neutralize_tokens` - (**bool, default=True**)
#   An indicator attribute to use masking for the computation of Blue and RougeL metrics. If True, counterfactual responses are masked using `CounterfactualGenerator.neutralize_tokens` method before computing the aforementioned metrics.
#
#   **Methods:**
#
#   1. `evaluate()` - Calculates counterfactual metrics for two sets of counterfactual outputs.
#       Method Parameters:
#
#       - `texts1` - (**List of strings**) A list of generated output from an LLM with mention of a protected attribute group.
#       - `texts2` - (**List of strings**) A list of equal length to `texts1` containing counterfactually generated output from an LLM with mention of a different protected attribute group.
#
#       Returns:
#       - A dictionary containing all Counterfactual metric values (**dict**).

counterfactual = CounterfactualMetrics()


similarity_values = {}
keys_, count = [], 1
for group1, group2 in combinations(["white", "black", "asian", "hispanic"], 2):
    keys_.append(f"{group1}-{group2}")
    similarity_values[keys_[-1]] = counterfactual.evaluate(
        race_eval_df[group1 + "_response"],
        race_eval_df[group2 + "_response"],
        attribute="race",
    )
    print(f"{count}. {group1}-{group2}")
    for key_ in similarity_values[keys_[-1]]:
        print("\t- ", key_, ": {:1.5f}".format(similarity_values[keys_[-1]][key_]))
    count += 1

# %%
# Next, we create a scatter plot to compare the metrics for different race combinations.
# Note: `matplotlib` installation is necessary to recreate the plot.

# Run this cell, if matplotlib is not installed. Install a pip package in the current Jupyter kernel
# import sys
# !{sys.executable} -m pip install matplotlib

import matplotlib.pyplot as plt

x = [x_ for x_ in range(6)]
fig, ax = plt.subplots()
for key_ in [
    "Cosine Similarity",
    "RougeL Similarity",
    "Bleu Similarity",
    "Sentiment Bias",
]:
    y = []
    for race_combination in similarity_values.keys():
        y.append(similarity_values[race_combination][key_])
    ax.scatter(x, y, label=key_)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
ax.set_ylabel("Metric Values")
ax.set_xlabel("Race Combinations")
ax.set_xticks(x)
ax.set_xticklabels(keys_, rotation=45)
plt.grid()
plt.show()


# %%
# .. _separate:
# 3.2 Separate Implementation
#
# 3.2.1 Counterfactual Sentiment Bias
#
# ``SentimentBias()`` - For calculating the counterfactual sentiment bias metric (class)
#
# **Class Attributes:**
#   - ``classifier`` - (**{'vader','NLP API'}**) Specifies which sentiment classifier to use. Currently, only vader is offered. `NLP API` coming soon.
#   - ``sentiment`` - (**{'neg','pos'}**) Specifies whether the classifier should predict positive or negative sentiment.
#   - ``parity`` - (**{'strong','weak'}, default='strong'**) Indicates whether to calculate strong demographic parity using Wasserstein-1 distance on score distributions or weak demographic parity using binarized sentiment predictions. The latter assumes a threshold for binarization that can be customized by the user with the `thresh` parameter.
#   - ``thresh`` - (**float between 0 and 1, default=0.5**) Only applicable if `parity` is set to 'weak', this parameter specifies the threshold for binarizing predicted sentiment scores.
#   - ``how`` : (**{'mean','pairwise'}, default='mean'**) Specifies whether to return the mean cosine similarity over all counterfactual pairs or a list containing cosine distance for each pair.
#   - ``custom_classifier`` - (**class object**) A user-defined class for sentiment classification that contains a `predict` method. The `predict` method must accept a list of strings as an input and output a list of floats of equal length. If provided, this takes precedence over `classifier`.
#
# **Methods:**
#
# 1. ``evaluate()`` - Calculates counterfactual sentiment bias for two sets of counterfactual outputs.
#
#   Method Parameters:
#
#     - ``texts1`` - (**List of strings**) A list of generated output from an LLM with mention of a protected attribute group
#     - ``texts2`` - (**List of strings**) A list of equal length to `texts1` containing counterfactually generated output from an LLM with mention of a different protected attribute group
#
#     Returns:
#     - Counterfactual Sentiment Bias score (**float**)

sentimentbias = SentimentBias()

# Sentiment Bias evaluation for race.

for group1, group2 in combinations(["white", "black", "asian", "hispanic"], 2):
    similarity_values = sentimentbias.evaluate(
        race_eval_df[group1 + "_response"], race_eval_df[group2 + "_response"]
    )
    print(
        f"{group1}-{group2} Strict counterfactual sentiment parity: ", similarity_values
    )

# %%
# 3.2.2 Cosine Similarity
#
# ``CosineSimilarity()`` - For calculating the social group substitutions metric (class)
#
# **Class Attributes:**
#
#   - ``SentenceTransformer`` - (**sentence_transformers.SentenceTransformer.SentenceTransformer, default=None**) Specifies which huggingface sentence transformer to use when computing cosine distance. See https://huggingface.co/sentence-transformers?sort_models=likes#models for more information. The recommended sentence transformer is 'all-MiniLM-L6-v2'.
#   - ``how`` - (**{'`mean`','`pairwise`'} default='mean'**) Specifies whether to return the mean cosine distance value over all counterfactual pairs or a list containing consine distance for each pair.
#
# **Methods:**
#
# 1. ``evaluate()`` - Calculates social group substitutions using cosine similarity. Sentence embeddings are calculated with `self.transformer`.
#
#   Method Parameters:
#
#     - ``texts1`` - (**List of strings**) A list of generated output from an LLM with mention of a protected attribute group
#     - ``texts2`` - (**List of strings**) A list of equal length to `texts1` containing counterfactually generated output from an LLM with mention of a different protected attribute group
#
#     Returns:
#     - Cosine distance score(s) (**float or list of floats**)

cosine = CosineSimilarity(transformer="all-MiniLM-L6-v2")

for group1, group2 in combinations(["white", "black", "asian", "hispanic"], 2):
    similarity_values = cosine.evaluate(
        race_eval_df[group1 + "_response"], race_eval_df[group2 + "_response"]
    )
    print(f"{group1}-{group2} Counterfactual Cosine Similarity: ", similarity_values)

# %%
# 3.2.3 RougeL Similarity
#
# ``RougeLSimilarity()`` - For calculating the social group substitutions metric using RougeL similarity (class)
#
# **Class Attributes:**
#   - ``rouge_metric`` : (**{`'rougeL'`,`'rougeLsum'`}, default='rougeL'**) Specifies which ROUGE metric to use. If sentence-wise assessment is preferred, select 'rougeLsum'.
#   - ``how`` - (**{`'mean'`,`'pairwise'`} default='mean'**) Specifies whether to return the mean cosine distance value over all counterfactual pairs or a list containing consine distance for each pair.
#
# **Methods:**
#
#  1. ``evaluate()`` - Calculates social group substitutions using ROUGE-L.
#
#   Method Parameters:
#
#     - ``texts1`` - (**List of strings**) A list of generated output from an LLM with mention of a protected attribute group
#     - ``texts2`` - (**List of strings**) A list of equal length to `texts1` containing counterfactually generated output from an LLM with mention of a different protected attribute group
#
#     Returns:
#     - ROUGE-L or ROUGE-L sums score(s) (**float or list of floats**)

rougel = RougelSimilarity()

for group1, group2 in combinations(["white", "black", "asian", "hispanic"], 2):
    # Neutralize tokens for apples to apples comparison
    group1_texts = cdg.neutralize_tokens(
        race_eval_df[group1 + "_response"], attribute="race"
    )
    group2_texts = cdg.neutralize_tokens(
        race_eval_df[group2 + "_response"], attribute="race"
    )

    # Compute and print metrics
    similarity_values = rougel.evaluate(group1_texts, group2_texts)
    print(f"{group1}-{group2} Counterfactual RougeL Similarity: ", similarity_values)

# %%
# 3.2.4 BLEU Similarity
#
# ``Bleu Similarity()`` - For calculating the social group substitutions metric using BLEU similarity (class)
#
# **Class parameters:**
#   - `how` - (**{'mean','pairwise'} default='mean'**) Specifies whether to return the mean cosine distance value over all counterfactual pairs or a list containing consine distance for each pair.
#
# **Methods:**
#
# 1. `evaluate()` - Calculates social group substitutions using BLEU metric.
#     Method Parameters:
#
#     - `texts1` - (**List of strings**) A list of generated output from an LLM with mention of a protected attribute group
#     - `texts2` - (**List of strings**) A list of equal length to `texts1` containing counterfactually generated output from an LLM with mention of a different protected attribute group
#
#     Returns:
#     - BLEU score(s) (**float or list of floats**)

bleu = BleuSimilarity()

for group1, group2 in combinations(["white", "black", "asian", "hispanic"], 2):
    # Neutralize tokens for apples to apples comparison
    group1_texts = cdg.neutralize_tokens(
        race_eval_df[group1 + "_response"], attribute="race"
    )
    group2_texts = cdg.neutralize_tokens(
        race_eval_df[group2 + "_response"], attribute="race"
    )

    # Compute and print metrics
    similarity_values = bleu.evaluate(group1_texts, group2_texts)
    print(f"{group1}-{group2} Counterfactual BLEU Similarity: ", similarity_values)


# %%
# .. _metric-defns:
# 4. Metric Definitions
# ---------------------
#
# Below are details of the LLM bias / fairness evaluation metrics calculated by the `CounterfactualMetrics` class. Metrics are defined in the context of a sample of :math:`N` LLM outputs, denoted :math:`\hat{Y}_1,...,\hat{Y}_N`. **Below, a  ❗ is used to indicate the metrics we deem to be of particular importance.**
#
# Counterfactual Fairness Metrics
# -------------------------------
#
# Given two protected attribute groups :math:`G', G''`, a counterfactual input pair is defined as a pair of prompts, :math:`X_i', X_i''` that are identical in every way except the former mentions protected attribute group :math:`G'` and the latter mentions :math:`G''`. Counterfactual metrics are evaluated on a sample of counterfactual response pairs :math:`(\hat{Y}_1', \hat{Y}_1''),...,(\hat{Y}_N', \hat{Y}_N'')` generated by an LLM from a sample of counterfactual input pairs :math:`(X_1',X_1''),...,(X_N',X_N'')`.
#
# *Counterfactual Similarity Metrics*
#
# Counterfactual similarity metrics assess similarity of counterfactually generated outputs. For the below three metrics, **values closer to 1 indicate greater fairness.**
#
# Counterfactual ROUGE-L (CROUGE-L) ❗
#
# CROUGE-L is defined as the average ROUGE-L score over counterfactually generated output pairs:
#
# .. math::
#   CROUGE\text{-}L =  \frac{1}{N} \sum_{i=1}^N \frac{2r_i'r_i''}{r_i' + r_i''},
# where
#
# .. math::
#   r_i' = \frac{LCS(\hat{Y}_i', \hat{Y}_i'')}{len (\hat{Y}_i') }, \quad r_i'' = \frac{LCS(\hat{Y}_i'', \hat{Y}_i')}{len (\hat{Y}_i'') }
#
# where :math:`LCS(\cdot,\cdot)` denotes the longest common subsequence of tokens between two LLM outputs, and :math:`len (\hat{Y})` denotes the number of tokens in an LLM output. The CROUGE-L metric effectively uses ROUGE-L to assess similarity as the longest common subsequence (LCS) relative to generated text length. For more on interpreting ROUGE-L scores, refer to `Klu.ai documentation <https://klu.ai/glossary/rouge-score#:~:text=A%20good%20ROUGE%20score%20varies,low%20at%200.3%20to%200.4.>`_
#
# Counterfactual BLEU (CBLEU)  ❗
#
# CBLEU is defined as the average BLEU score over counterfactually generated output pairs:
#
# .. math::
#   CBLEU =  \frac{1}{N} \sum_{i=1}^N \min(BLEU(\hat{Y}_i', \hat{Y}_i''), BLEU(\hat{Y}_i'', \hat{Y}_i')).
# For more on interpreting BLEU scores, refer to `Google's documentation <https://cloud.google.com/translate/automl/docs/evaluate>`_.
#
# Counterfactual Cosine Similarity (CCS)  ❗
#
# Given a sentence transformer :math:`\mathbf{V} : \mathcal{Y} \xrightarrow{} \mathbb{R}^d`, CCS is defined as the average cosine simirity score over counterfactually generated output pairs:
#
# .. math::
#   CCS = \frac{1}{N} \sum_{i=1}^N   \frac{\mathbf{V}(Y_i') \cdot \mathbf{V}(Y_i'') }{ \lVert \mathbf{V}(Y_i') \rVert \lVert \mathbf{V}(Y_i'') \rVert},
#
# *Counterfactual Sentiment Metrics*
#
# Counterfactual sentiment metrics leverage a pre-trained sentiment classifier :math:`Sm: \mathcal{Y} \xrightarrow[]{} [0,1]` to assess sentiment disparities of counterfactually generated outputs. For the below three metrics, **values closer to 0 indicate greater fairness.**
# Counterfactual Sentiment Bias (CSB)  ❗
#
# CSP calculates Wasserstein-1 distance \citep{wasserstein} between the output distributions of a sentiment classifier applied to counterfactually generated LLM outputs:
#
# .. math::
#   CSP = \mathbb{E}_{\tau \sim \mathcal{U}(0,1)} | P(Sm(\hat{Y}') > \tau) -  P(Sm(\hat{Y}'') > \tau)|,
# where :math:`\mathcal{U}(0,1)` denotes the uniform distribution. Above, :math:`\mathbb{E}_{\tau \sim \mathcal{U}(0,1)}` is calculated empirically on a sample of counterfactual response pairs :math:`(\hat{Y}_1', \hat{Y}_1''),...,(\hat{Y}_N', \hat{Y}_N'')` generated by :math:`\mathcal{M}`, from a sample of counterfactual input pairs :math:`(X_1',X_1''),...,(X_N',X_N'')` drawn from :math:`\mathcal{P}_{X|\mathcal{A}}`.
