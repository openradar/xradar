# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

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
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `xradar` for local development.

1. Fork the `xradar` repo on GitHub.
2. Clone your fork locally:

   ```bash
   $ git clone git@github.com:your_name_here/xradar.git
   ```

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:

   ```bash
   $ mkvirtualenv xradar
   $ cd xradar/
   $ python -m pip install -e .[dev]
   ```

   * Install `ffmpeg` using your OS package manager.

4. Create a branch for local development:

   ```bash
   $ git checkout -b name-of-your-bugfix-or-feature
   ```
   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass `black`, `ruff` and the
   tests:

   ```bash
   $ black --check .
   $ ruff check .
   $ python -m pytest
   ```
   To get `black` and `ruff`, just pip install them into your virtualenv.

6. Install pre-commit hooks

   We highly recommend that you setup [pre-commit](https://pre-commit.com/) hooks to automatically
   run all the above tools (and more) every time you make a git commit. To install the hooks:

   ```bash
   $ python -m pip install pre-commit
   $ pre-commit install
   ```
   To run unconditionally against all files:

   ```bash
   $ pre-commit run --all-files
   ```
   You can skip the pre-commit checks with ``git commit --no-verify``.

7. Commit your changes and push your branch to GitHub:

   ```bash
   $ git add .
   $ git commit -m "Your detailed description of your changes."
   $ git push origin name-of-your-bugfix-or-feature
   ```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for [supported Python versions](https://endoflife.date/python) and for PyPy. Check
   [GHA](https://github.com/openradar/xradar/actions)
   and make sure that the tests pass for all supported Python versions.

## Building the documentation

To build the documentation run:

   ```bash
   $ cd docs
   $ make html
   ```
Then you can access the documentation via browser locally by opening `docs/_build/html/index.html`.

## Tips

To run a subset of tests:

   ```bash
   $ pytest tests.test_xradar
   ```

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in [history.md](https://github.com/openradar/xradar/blob/main/docs/history.md)).
Then just create a release in the GitHub workflow. GHA will then deploy to PyPI if tests pass.
