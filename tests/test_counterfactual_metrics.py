import json

import numpy as np

from langfair.metrics.counterfactual import CounterfactualMetrics
from langfair.metrics.counterfactual.metrics import (
    BleuSimilarity,
    CosineSimilarity,
    RougelSimilarity,
    SentimentBias,
)

datafile_path = "tests/data/counterfactual/counterfactual_data_file.json"
with open(datafile_path, "r") as f:
    data = json.load(f)

actual_result_file_path = "tests/data/counterfactual/counterfactual_results_file.json"
with open(actual_result_file_path, "r") as f:
    actual_results = json.load(f)

def test_bleu():
    bleu = BleuSimilarity()
    x = bleu.evaluate(data["text1"], data["text2"])
    np.testing.assert_almost_equal(x, actual_results["test1"], 5)

# def test_cosine():
#     cosine = CosineSimilarity(transformer='all-MiniLM-L6-v2')
#     x = cosine.evaluate(data["text1"], data["text2"])
#     np.testing.assert_almost_equal(x, actual_results["test2"], 5)

def test_rougel():
    rougel = RougelSimilarity()
    assert rougel.evaluate(data["text1"], data["text2"]) == actual_results["test3"]

def test_senitement1():
    sentiment = SentimentBias()
    assert sentiment.evaluate(data["text1"], data["text2"]) == actual_results["test4"]

def test_senitement2():
    sentiment = SentimentBias(parity="weak")
    assert sentiment.evaluate(data["text1"], data["text2"]) == actual_results["test5"]

def test_CounterfactualMetrics():
    metrics = [# "Cosine", 
               "Rougel", "Bleu", "Sentiment Bias"
    ]
    counterfactualmetrics = CounterfactualMetrics(metrics=metrics)
    score = counterfactualmetrics.evaluate(data["text1"], data["text2"], attribute="race")
    ans = actual_results["test6"]
    assert all([
        abs(score[key]-ans[key])<1e-5 for key in ans
        if key != "Cosine Similarity"
    ])
