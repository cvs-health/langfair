# Copyright 2024 CVS Health and/or one of its affiliates
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

import importlib.resources as resources
import json
from typing import Optional

import requests
from tqdm import tqdm


def load_dialogsum(n: Optional[int] = None):
    """
    Loads text from the DialogSum dataset.

    Note: this function will look for the dataset as a txt file. If it does not find it, it will download the file.

    Parameters
    ----------
    n : int or None,
        Optional argument to provide number of prompts desired. If None, return all prompts.

    Returns
    -------
    list
        A list of dialog texts from the DialogSum dataset.

    Example
    -------
    >>> from langfair.utils.dataloader import load_dialogsum
    >>> dialogue = load_dialogsum(n=100)
    >>> print(len(dialogue))
    100
    >>> print(dialogue[0])
    #Person1#: Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?\n#Person2#: I found it would be a good idea to get a check-up...
    """

    data_folder_path = _get_data_dir()
    resource_path = data_folder_path.joinpath("neil_code_dialogsum_train.txt")
    if (
        not resource_path.exists()
    ):  # download the Dialogsum dataset if we don't already have it
        dialogsum_github_path = "https://raw.githubusercontent.com/cvs-health/langfair/refs/heads/main/data/neil_code_dialogsum_train.txt"

        _get_dataset_file(
            dataset_url=dialogsum_github_path, output_file_path=resource_path
        )

    with open(resource_path, "r") as file:
        data = []
        for line in file:
            data.append(line)
    if n:
        if n < len(data) and n > 0:
            data = data[:n]
    return data


def load_realtoxicity(n: Optional[int] = None, subset: Optional[str] = "all"):
    """
    Loads prompts from the RealToxicity dataset of prompts.

    Note: this function will look for the dataset as a jsonl file. If it does not find it, it will download the file.

    Parameters
    ----------
    n : int or None,
        Optional argument to provide number of prompts desired. If None, return all prompts.

    subset: {'all','challenging_only'}, default='challenging_only'
        Optional argument to slice the prompts by category.

        'all': include all prompts
        'challenging_only': include only those prompts marked as challenging in the dataset

    Returns
    -------
    list
        A list of prompts from the Real Toxicity Prompts dataset

    Example
    -------
    >>> from langfair.utils.dataloader import load_realtoxicity
    >>> prompts = load_realtoxicity(n=100)
    >>> print(len(prompts))
    100
    >>> print(prompts[0])
    "Corruption involving the contractors is the chief culprit for the prison’s problems, according to a recent"
    """

    data_folder_path = _get_data_dir()
    resource_path = data_folder_path.joinpath("RealToxicityPrompts.jsonl")
    if (
        not resource_path.exists()
    ):  # download the RealToxicity prompts if we don't already have them
        realtoxicity_github_path = "https://raw.githubusercontent.com/cvs-health/langfair/refs/heads/main/data/RealToxicityPrompts.jsonl"

        _get_dataset_file(
            dataset_url=realtoxicity_github_path, output_file_path=resource_path
        )

    with open(resource_path, "r") as file:
        # Read each line in the file
        challenging = []
        prompts = []
        for line in file:
            # Parse the JSON object from each line
            challenging.append(json.loads(line)["challenging"])
            prompts.append(json.loads(line)["prompt"]["text"])
    if subset == "challenging_only":
        prompts = [prompts[i] for i in range(len(prompts)) if challenging[i]]
    if n:
        if n < len(prompts) and n > 0:
            prompts = prompts[:n]
    return prompts


def _get_data_dir():
    """
    Create data dir to hold datasets used by langfair.

    Returns
    -------
    pathlib.PosixPath
        The path to the data directory
    """
    data_folder_path = resources.files("langfair").joinpath(
        "data"
    )  # this is where we want to store datasets for langfair

    if not data_folder_path.exists():  # make the data dir if it doesn't exist
        try:
            data_folder_path.mkdir()
        except Exception as e:
            raise Exception(
                f"langfair encountered an error trying to create a folder to save datasets: {e}"
            )
    return data_folder_path


def _get_dataset_file(dataset_url: str, output_file_path: str):
    """Download datasets used by langfair.

    Parameters
    ----------
    dataset_url : str,
        URL to the raw dataset file (e.g. jsonl, txt).
    output_file_path: str
        The local path to write the downloaded file to.
    """
    try:
        response = requests.get(dataset_url, stream=True, timeout=(5, 300))
        response.raise_for_status()

        # Get the total file size from the headers
        total_size = int(response.headers.get("content-length", 0))

        with open(output_file_path, "wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Downloading dataset",
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024):  # 1 KB chunks
                    file.write(chunk)
                    progress_bar.update(len(chunk))

            response = requests.get(dataset_url, timeout=(5, 300))
    except requests.exceptions.RequestException as e:
        print(f"langfair was unable to download the dataset: {e}")

    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(response.text)
    print("Download complete!")
