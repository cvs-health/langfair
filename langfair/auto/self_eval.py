# Copyright 2026 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

from langfair.auto.base import MetricTypes, _BaseEval
from langfair.constants.cost_data import FAILURE_MESSAGE
from langfair.generator import CounterfactualGenerator
from langfair.utils.display import start_progress_bar


class SelfEval(_BaseEval):
    def __init__(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None,
        count: int = 25,
        metrics: MetricTypes = None,
        counterfactual_transformer: str = "all-MiniLM-L6-v2",
        counterfactual_sentiment_classifier: str = "vader",
        toxicity_device: str = "cpu",
        neutralize_tokens: bool = True,
        _stereotype_classifier_model: str = "wu981526092/Sentence-Level-Stereotype-Detector",
    ) -> None:
        """
        This class calculates toxicity, stereotype, and counterfactual metrics with
        response generation delegated to the user, so that no LangChain LLM object
        is required. It implements a two-phase workflow:

        Phase 1 (prompt preparation, steps 1-2): on construction, prompts are
        checked for protected attribute words (fairness through unawareness check).
        The first call to `get_prompts()` then creates the counterfactual prompt
        variants and returns a flat, ordered list of every prompt for which the
        user must generate a response (each prompt is repeated `count` times).

        Step 3, response generation, is delegated to the user: exactly one response
        per element of `get_prompts()`, produced with their own stack and preserving
        order.

        Phase 2 (steps 4-6): those responses are passed to
        `evaluate(responses=...)`, which computes the same metrics as
        `AutoEval.evaluate` under the same step numbering.

        Parameters
        ----------
        prompts : list of strings
            A list of input prompts for the model.

        responses : list of strings, default=None
            A list of generated output from an LLM, corresponding element-wise to
            `prompts`. If provided, `get_prompts()` returns only the counterfactual
            prompt variants still requiring generation.

        count : int, default=25
            Specifies number of responses the user should generate for each prompt.
            The convention is to use 25 generations per prompt in evaluating toxicity.
            See, for example DecodingTrust (https://arxiv.org/abs//2306.11698)
            or Gehman et al., 2020 (https://aclanthology.org/2020.findings-emnlp.301/).

        metrics : dict or list of str, default option compute all supported metrics.
            Specifies which metrics to evaluate. Note that `metrics` is fixed at
            construction, since the prompt list returned by `get_prompts()` depends
            on whether counterfactual metrics are requested.

        counterfactual_transformer: str, default="all-MiniLM-L6-v2"
            Specifies which huggingface sentence transformer to use when computing cosine distance. See
            https://huggingface.co/sentence-transformers?sort_models=likes#models
            for more information. The recommended sentence transformer is 'all-MiniLM-L6-v2'. User can also specify a local path to a model.

        counterfactual_sentiment_classifier: str, default="vader"
            Specifies the sentiment classifier to use for counterfactual sentiment bias calculation.

        toxicity_device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that toxicity classifiers use for prediction. Set to "cuda" for classifiers to be able
            to leverage the GPU. Currently, 'detoxify_unbiased' and 'detoxify_original' will use this parameter.

        neutralize_tokens: boolean, default=True
            An indicator attribute to use masking for the computation of Blue and RougeL metrics. If True, counterfactual
            responses are masked using `CounterfactualGenerator.neutralize_tokens` method before computing the aforementioned metrics.
        """
        super().__init__(
            prompts=prompts,
            responses=responses,
            metrics=metrics,
            counterfactual_transformer=counterfactual_transformer,
            counterfactual_sentiment_classifier=counterfactual_sentiment_classifier,
            toxicity_device=toxicity_device,
            neutralize_tokens=neutralize_tokens,
            _stereotype_classifier_model=_stereotype_classifier_model,
        )
        if not isinstance(count, int) or count < 1:
            raise ValueError("`count` must be a positive integer.")
        if self.responses is not None and len(self.responses) != len(self.prompts):
            raise ValueError(
                "When `responses` are provided, they must have the same length as `prompts` "
                f"(received {len(self.responses)} responses for {len(self.prompts)} prompts)."
            )
        self.count = count
        self.cf_generator_object = CounterfactualGenerator()

        # Step 1: Fairness Through Unawareness check (no LLM required)
        self.protected_words, self.total_protected_words = self._check_ftu(
            show_progress_bars=False
        )
        if self.total_protected_words > 0:
            print(
                "Fairness through unawareness is not satisfied. Counterfactual prompts will be included."
            )
        else:
            print("FTU is satisfied. Counterfactual assessment will be skipped.")

        # Step 2 (counterfactual dataset generation) is deferred to `get_prompts`
        self._build_counterfactual = (
            "counterfactual" in self.metrics and self.total_protected_words > 0
        )
        self._cf_prompts_dicts: Dict[str, Dict[str, List[str]]] = {}
        self._segments: List[Dict[str, Any]] = []
        self._segments_built = False

    def get_prompts(self) -> List[str]:
        """
        Perform step 2 (counterfactual dataset generation) on first call, then
        return the flat, ordered list of prompts for which the user must generate
        responses before calling `evaluate`.

        The list is ordered as follows: the original prompts first (omitted if
        `responses` was provided to the constructor), followed by the counterfactual
        prompt variants for each protected attribute found in the prompts (race
        before gender), group by group ('white', 'black', 'hispanic', 'asian' for
        race; 'male', 'female' for gender). Each prompt appears `count` times
        consecutively. Use the `prompt_manifest` property to inspect the segment
        boundaries.

        Returns
        -------
        list of str
            Prompts to generate responses for. The responses passed to `evaluate`
            must satisfy `responses[i]` being a response to `get_prompts()[i]`.
        """
        self._ensure_segments(verbose=True)
        return list(self._expanded_prompts)

    @property
    def prompt_manifest(self) -> List[Dict[str, Any]]:
        """
        Read-only breakdown of the segments composing `get_prompts()`. Each element
        describes one segment with keys 'kind' ('original' or 'counterfactual'),
        'attribute', 'group', 'start', 'end' (index range in the flat list),
        'n_unique_prompts', and 'count'.
        """
        self._ensure_segments()
        return [
            {
                "kind": segment["kind"],
                "attribute": segment["attribute"],
                "group": segment["group"],
                "start": segment["start"],
                "end": segment["end"],
                "n_unique_prompts": len(segment["unique_prompts"]),
                "count": self.count,
            }
            for segment in self._segments
        ]

    def evaluate(
        self,
        responses: Optional[List[str]] = None,
        return_data: bool = False,
        show_progress_bars: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute all the metrics based on user-provided responses. This method makes
        no LLM calls and runs synchronously.

        Parameters
        ----------
        responses : list of str, default=None
            One response per element of `get_prompts()`, in the same order, i.e.
            `responses[i]` must be a response to `get_prompts()[i]`. For any
            generation that failed, insert the literal failure message
            'Unable to get response' (`langfair.constants.cost_data.FAILURE_MESSAGE`);
            counterfactual response pairs containing it are excluded from
            counterfactual metrics. May be omitted only when `get_prompts()` is
            empty (i.e., `responses` were provided to the constructor and no
            counterfactual generations are required).

        return_data : bool, default=False
            Indicates whether to include response-level scores in results dictionary returned by this method.

        show_progress_bars : bool, default=True
            If True, displays progress bars while computing metrics.

        Returns
        -------
        dict
            A dictionary containing values of toxicity, stereotype, and counterfactual metrics and, optionally,
            response-level scores.
        """
        self._ensure_segments()
        responses = [] if responses is None else list(responses)
        if len(responses) != len(self._expanded_prompts):
            raise ValueError(self._length_error_message(len(responses)))
        for i, response in enumerate(responses):
            if not isinstance(response, str):
                raise TypeError(
                    f"All responses must be strings; received {type(response).__name__} at index {i}."
                )

        self._ingest_responses(responses)

        if show_progress_bars:
            self.progress_bar = start_progress_bar()

        # Step 1 (FTU check) runs at construction, step 2 (counterfactual datasets)
        # in `get_prompts`, and step 3 (response generation) is performed by the
        # user, so this method picks up the same step numbering as
        # `AutoEval.evaluate` at step 4.
        # 4. Calculate toxicity metrics
        self._evaluate_toxicity(show_progress_bars)

        # 5. Calculate stereotype metrics
        self._evaluate_stereotype(self.protected_words, show_progress_bars)

        # 6. Calculate CF metrics (if FTU not satisfied and counterfactual metrics requested)
        self._evaluate_counterfactual(
            self.protected_words,
            self.total_protected_words,
            show_progress_bars,
        )

        return self._finalize(return_data, show_progress_bars)

    def _ensure_segments(self, verbose: bool = False) -> None:
        """
        Run step 2 (counterfactual dataset generation) on first call and no-op
        thereafter. Deferred from construction so that the step is reported when
        the user asks for the prompts via `get_prompts`, while `prompt_manifest`
        and `evaluate` can still build the segments silently if `get_prompts` was
        never called.
        """
        if self._segments_built:
            return
        if verbose:
            if self._build_counterfactual:
                print("Step 2: Generate Counterfactual Datasets")
                print("----------------------------------------")
            else:
                print("(Skipping) Step 2: Generate Counterfactual Datasets")
                print("---------------------------------------------------")
        self._build_segments()
        self._segments_built = True

    def _build_segments(self) -> None:
        """
        Construct the ordered prompt segments defining the `get_prompts()` contract:
        the original prompts (unless `responses` was provided at construction),
        then, for each protected attribute with attribute words present (when
        counterfactual metrics are requested), one segment per counterfactual
        group. Each prompt is expanded `count` times, consecutively.
        """
        cursor = 0

        def add_segment(
            kind: str,
            attribute: Optional[str],
            group: Optional[str],
            unique_prompts: List[str],
        ) -> None:
            nonlocal cursor
            expanded = [prompt for prompt in unique_prompts for _ in range(self.count)]
            self._segments.append(
                {
                    "kind": kind,
                    "attribute": attribute,
                    "group": group,
                    "unique_prompts": list(unique_prompts),
                    "expanded_prompts": expanded,
                    "start": cursor,
                    "end": cursor + len(expanded),
                }
            )
            cursor += len(expanded)

        if self.responses is None:
            add_segment("original", None, None, self.prompts)

        if self._build_counterfactual:
            for attribute in self.protected_words.keys():
                if self.protected_words[attribute] > 0:
                    prompts_dict = self.cf_generator_object.create_prompts(
                        prompts=self.prompts, attribute=attribute
                    )
                    self._cf_prompts_dicts[attribute] = prompts_dict
                    for group in self.cf_generator_object.group_mapping[attribute]:
                        add_segment(
                            "counterfactual",
                            attribute,
                            group,
                            prompts_dict[group + "_prompt"],
                        )

        self._expanded_prompts = [
            prompt
            for segment in self._segments
            for prompt in segment["expanded_prompts"]
        ]

    def _ingest_responses(self, responses: List[str]) -> None:
        """
        Split the flat, ordered response list back into the per-segment structures
        consumed by the metric computation steps: `self.responses` (with
        `self.prompts` expanded to match) and `self.counterfactual_responses` in the
        same format returned by `CounterfactualGenerator.generate_responses`.
        """
        cf_prompts: Dict[str, Dict[str, List[str]]] = {}
        cf_responses: Dict[str, Dict[str, List[str]]] = {}
        for segment in self._segments:
            segment_responses = responses[segment["start"] : segment["end"]]
            if segment["kind"] == "original":
                self.prompts = list(segment["expanded_prompts"])
                self.responses = segment_responses
            else:
                attribute = segment["attribute"]
                cf_prompts.setdefault(attribute, {})[segment["group"]] = segment[
                    "expanded_prompts"
                ]
                cf_responses.setdefault(attribute, {})[segment["group"]] = (
                    segment_responses
                )

        self.counterfactual_responses = None
        if cf_responses:
            self.counterfactual_responses = {}
            for attribute in cf_responses:
                groups = self.cf_generator_object.group_mapping[attribute]
                all_responses = [
                    response
                    for group in groups
                    for response in cf_responses[attribute][group]
                ]
                n_failed = sum(
                    1 for response in all_responses if response == FAILURE_MESSAGE
                )
                self.counterfactual_responses[attribute] = {
                    "data": {
                        **{
                            f"{group}_prompt": cf_prompts[attribute][group]
                            for group in groups
                        },
                        **{
                            f"{group}_response": cf_responses[attribute][group]
                            for group in groups
                        },
                    },
                    "metadata": {
                        "non_completion_rate": (
                            n_failed / len(all_responses) if all_responses else 0.0
                        ),
                        "count": self.count,
                        "groups": list(groups),
                        "original_prompts": self._cf_prompts_dicts[attribute][
                            "original_prompt"
                        ],
                        "attribute_words": self._cf_prompts_dicts[attribute][
                            "attribute_words"
                        ],
                    },
                }

    def _length_error_message(self, received: int) -> str:
        """Build the error message for a response list of the wrong length."""
        lines = [
            f"SelfEval.evaluate expected {len(self._expanded_prompts)} responses but received {received}.",
            "Expected response order (from `get_prompts()`):",
        ]
        for segment in self._segments:
            label = (
                "original prompts"
                if segment["kind"] == "original"
                else f"counterfactual {segment['attribute']}/{segment['group']} prompts"
            )
            lines.append(
                f"  [{segment['start']}:{segment['end']}] {label} - "
                f"{len(segment['unique_prompts'])} prompts x count {self.count}"
            )
        lines.append(
            "Generate one response per element of `get_prompts()`, preserving order. "
            f"Use '{FAILURE_MESSAGE}' for any generation that failed."
        )
        return "\n".join(lines)
