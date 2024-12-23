# Contributing to LangFair

Welcome and thank you for considering contributing to LangFair!

It takes a lot of time and effort to use software much less build upon it, so we deeply appreciate your desire to help make this project thrive.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Pull Requests](#pull-requests)
2. [Development Setup](#development-setup)
3. [Style Guides](#style-guides)
    - [Code Style](#code-style)
4. [License](#license)

## How to Contribute

### Reporting Bugs

If you find a bug, please report it by opening an issue on GitHub. Include as much detail as possible:
- Steps to reproduce the bug.
- Expected and actual behavior.
- Screenshots if applicable.
- Any other information that might help us understand the problem.

### Suggesting Enhancements

We welcome suggestions for new features or improvements. To suggest an enhancement, please open an issue on GitHub and include:
- A clear description of the suggested enhancement.
- Why you believe this enhancement would be useful.
- Any relevant examples or mockups.

### Pull Requests

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a pull request.

Please ensure your pull request adheres to the following guidelines:
- Follow the project's code style.
- Include tests for any new features or bug fixes.

## Development Setup

1. Clone the repository: `git clone https://github.aetna.com/cvs-health/langfair`
2. Navigate to the project directory: `cd langfair`
3. Create and activate a virtual environment (using `venv` or `conda`)
4. Install dependencies: `poetry install`
5. Install our pre-commit hooks to ensure code style compliance: `pre-commit install`
6. Run tests to ensure everything is working: `pre-commit run --all-files`

You're ready to develop!

**For documentation contributions**
Our documentation lives on the gh-pages branch and is hosted via GitHub Pages.

There are two relevant directories:
* `docs_src` - where source documentation files are located
* `docs` - where the built documentation that is shown on GitHub Pages lives.

To build documentation:
1. Checkout the `gh-pages` branch
2. Navigate to the source dir: `cd docs_src`
3. Build documentation for a GitHub Pages deployment: `make github`

To update docs site with new langfair release:
__(here we'll use an example scenario where langfair was at v0.2.0 and we are now rebuilding for a recent v0.3.0)__
1. Merge the release pr/branch/tag for v0.3.0 into the gh-pages branch
2. Make a duplicate copy of the `docs_src/latest` dir and rename it to the previous latest's release number (in this case "0.2")
3. Repeat step 2 for `docs/latest` dir
4. In `docs_src/versions.json` 
    - make a new entry for the new release (`name: "v0.3 (latest)"`)
    - remove "latest" from the name of v0.2
    - change the url for v0.2 to point at /0.2/index.html instead of /latest/index.thml
    - make sure `"preferred": true` is an attribute of the new v0.3 version only
5. Build latest (v0.3) docs by navigating to `docs_src/latest` and running `make github`
6. Push changes to your feature branch and verify successful update on your forked repo's deployed doc site before submitting a PR


## Style Guides

### Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) to lint and format our files.
- Our pre-commit hook will run Ruff linting and formatting when you commit.
- You can manually run Ruff at any time (see [Ruff usage](https://github.com/astral-sh/ruff#usage)).

Please ensure your code is properly formatted and linted before committing.

## License

Before contributing to this CVS Health sponsored project, you will need to sign the associated [Contributor License Agreement (CLA)](https://forms.office.com/r/gMNfs4yCck)

---

Thanks again for using and supporting LangFair!