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

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in [history.md](https://github.com/openradar/xradar/blob/main/docs/history.md)).
Then just create a release in the GitHub workflow. GHA will then deploy to PyPI if tests pass.
