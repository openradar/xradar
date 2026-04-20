# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

Report bugs at [xradar-issues](https://github.com/openradar/xradar/issues).

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

xradar could always use more documentation, whether as part of the
official xradar docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at [xradar-issues](https://github.com/openradar/xradar/issues).

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## AI Usage Policy

xradar contributors may use AI tools as part of their workflow. Please read the
[AI Usage Policy](https://github.com/openradar/xradar/blob/main/docs/ai_policy.md)
before submitting an AI-assisted contribution. In short: you are responsible
for every line of code in your PR, PR descriptions and review replies must be
your own words, and large AI-assisted changes — as well as CI, packaging, and
dependency changes — should start with an issue rather than a surprise diff.

## Get Started!

Ready to contribute? Here's how to set up `xradar` for local development.

1. Fork the `xradar` repo on GitHub.
2. Clone your fork locally.

   ```bash
   git clone git@github.com:your_name_here/xradar.git
   git remote add upstream https://github.com/openradar/xradar
   cd xradar
   ```

3. Install `xradar` development environment.

  - Install with conda (Linux):

    ```bash
    curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b
    rm Miniforge3-Linux-x86_64.sh
    source $HOME/miniforge3/etc/profile.d/conda.sh
    conda env create --file environment.yml
    conda activate xradar-dev
    ```

  - Install with conda (Windows):

    ```bash
    winget install -e --id CondaForge.Miniforge3
    & "$HOME\Miniforge3\shell\condabin\conda-hook.ps1"
    conda env create --file environment.yml
    conda activate xradar-dev
    ```

  - Install with uv (Linux):

    ```bash
    sudo apt install build-essential ffmpeg libhdf5-dev libnetcdf-dev pandoc
    sudo snap install astral-uv --classic
    uv python install 3.13
    uv venv --python 3.13
    source .venv/bin/activate
    uv pip install -e .[dev] --no-binary=netcdf4 --no-binary=h5py
    ```

  - Install with uv (Windows):

    ```powershell
    winget install ffmpeg
    winget install uv
    uv python install 3.13
    uv venv --python 3.13
    .venv\Scripts\Activate.ps1
    uv pip install -e .[dev]
    ```

4. Create a branch for local development.

    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```
    Now you can make your changes locally.

5. Perform tests and ensure your changes are covered.
    ```bash
    python -m pytest -k name_of_my_test
    python -m pytest -n auto --dist loadfile --cov=xradar --cov-report xml --verbose
    python -m coverage report
    diff-cover coverage.xml --compare-branch=main
    ```

6. Update documentation.
    - Add docstring to new functions
        ```bash
        python -m sphinx build -j auto -v -b html docs/ doc-build
        ```
    - Check at `docs/_build/html/index.html`
    - Add new features to README.rst

7. Install pre-commit hooks

    ```bash
    pre-commit install
    ```

8. Commit your changes and push your branch to GitHub.

    ```bash
    git add your-new-file.py
    git commit -a -m "Brief summary of the changes" -m " * Change A" -m " * Change B"
    git push origin name-of-your-bugfix-or-feature
    ```

9. Submit a pull request.

    - Go to your fork on GitHub
    - Submit your pull request as a draft
    - Verify all checks are passing
    - Mark your pull request as ready for review

## Roles and Responsibilities

We want contribution to xradar to have a clear path from first contribution to
shared project stewardship. The project aims to keep responsibility distributed,
make participation transparent, and create an approachable way for contributors
to become more involved over time.

### Contributors

Contributors improve xradar through code, documentation, bug reports, testing,
examples, and discussion. This is the starting point for everyone.

Typical ways contributors help include:

* opening focused issues with reproducible examples
* submitting pull requests
* improving documentation, examples, or tests
* helping with design discussions and user support

### Team Members

Team members are contributors who want to become more involved in the day-to-day
work of the project. In addition to regular contributions, they may help with:

* issue triage and labeling
* reproducing bug reports
* answering questions and guiding new contributors
* identifying pull requests that need review or follow-up

This role is a good way to get familiar with the project workflow and where help
is most useful.

### Maintainers

Maintainers are active, trusted contributors with broader responsibility for the
health of the project. Maintainers are expected to:

* review pull requests constructively and in a timely manner
* help guide technical discussions and decisions
* merge contributions when appropriate
* support contributors and encourage sustainable project practices
* help maintain release and documentation quality

Maintainer status reflects sustained engagement and judgment, not just the
number of contributions.

## Pathway to Greater Involvement

There is no strict checklist for moving between roles, but the general pathway is:

1. Start as a contributor by opening issues or pull requests.
2. Become a regular contributor by showing sustained involvement, reliability,
   or expertise in parts of the project.
3. Take on more community-facing work such as triaging issues, reviewing pull
   requests, or helping guide discussions.
4. Be invited to join the maintainer group based on continued engagement and
   project trust.

If you contribute regularly and would like to get more involved, reach out in an
issue, pull request, or discussion. We want the process to be welcoming and
transparent.

## Deploying

A reminder for maintainers on how to deploy.
Make sure all your changes are committed (including an entry in [history.md](https://github.com/openradar/xradar/blob/main/docs/history.md)).
Then just create a release in the GitHub workflow. GHA will then deploy to PyPI if tests pass.
