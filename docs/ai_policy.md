<!-- markdownlint-disable MD013 -->

# AI Usage Policy

Some xradar contributors and maintainers use AI tools as part of their development
workflow. We assume this is now common. Tools, patterns, and norms are evolving
fast — this policy aims to avoid restricting contributors' choice of tooling while
ensuring that:

- Reviewers are not overburdened
- Contributions can be maintained
- The submitter can vouch for and explain all changes
- Developers can acquire new skills[^skills]
- Contributions do not put the project's license at risk

This policy applies regardless of whether the code was written by hand, with AI
assistance, or generated entirely by an AI tool. It is adapted from the
[xarray AI Usage Policy](https://github.com/pydata/xarray/blob/main/doc/contribute/ai-policy.md)
and aligns with similar efforts across the scientific Python ecosystem, with
xradar-specific additions for CI, packaging, and dependency changes, copyright
and licensing, and disclosure of AI usage.

[^skills]:
    Over-reliance on AI tools has been shown to
    [hinder skill formation amongst software developers](https://arxiv.org/abs/2601.20245).

## Core Principle: Changes

If you submit a pull request, you are responsible for understanding and having
fully reviewed the changes. You must be able to explain why each change is
correct and how it fits into the project. Strive to minimize changes to ease the
burden on reviewers — avoid including unnecessary or loosely related changes.
AI tools tend to produce verbose code, tests, and text; more code is more code
to maintain, so keep contributions concise.

If you are unsure about the best way forward, open a draft PR and use it to
discuss the approach with maintainers before expanding the scope.

## Core Principle: Communication

PR descriptions, issue comments, and review responses must be your own words.
The substance and reasoning must come from you. Do not paste AI-generated text
as comments or review responses. Please attempt to be concise.

PR descriptions should follow the provided template.

Using AI to improve the language of your writing (grammar, phrasing, spelling,
etc.) is acceptable. Be careful that it does not introduce inaccurate details in
the process.

Maintainers reserve the right to delete or hide comments that violate our AI
policy or code of conduct.

## Code and Tests

### Review Every Line

You must have personally reviewed and understood all changes before submitting.

If you used AI to generate code, you are expected to have read it critically and
tested it. As with a hand-written PR, the description should explain the
approach and reasoning behind the changes. Do not leave it to reviewers to
figure out what the code does and why.

AI tools can also help you review your own changes, for example by running a
second, independent model over the diff. That can be a useful extra check, but
it does not replace your own review: you remain responsible for every line.

#### Not Acceptable

> I pointed an agent at the issue and here are the changes

> This is what Claude came up with. 🤷

#### Acceptable

> I iterated multiple times with an agent to produce this. The agent wrote the
> code at my direction, and I have fully read and validated the changes.

> I pointed an agent at the issue and it generated a first draft. I reviewed
> the changes thoroughly and understand the implementation well.

### Prefer Small PRs and Open an Issue First

Generating code with agents is fast and easy. Reviewing it is not. Any PR —
hundreds of lines or just tens — that is hard to review shifts the burden from
the contributor to the reviewer. Line count is not the only criterion; what
matters is how confidently a reviewer can follow the change end-to-end.

**Prefer small, focused PRs over one large one.** A series of small,
independently reviewable PRs — each tied to a clear issue — is almost always
easier to review, merge, and revert than a single larger change. If an
AI-assisted contribution can be decomposed into smaller pieces, please do so.

**Strongly encouraged: open an issue first** for any non-trivial AI-assisted
contribution (for example, a refactor, a framework migration, a new subsystem,
or any change whose review burden is non-obvious at a glance). The issue
should describe the motivation, the proposed approach, and the expected
scope. Maintainers will use it to confirm the work is a good fit for the
project, surface concerns early, and agree on how the change should be
structured **before any code is written**. This prevents the common AI-era
pattern of a surprise diff that nobody asked for and nobody wants to review.

Maintainers reserve the right to close PRs where the scope makes meaningful
review impractical, or when they suspect this policy has been violated.
Similarly they may request that large changes be broken into smaller,
reviewable pieces.

### CI, Packaging, and Dependency Changes

Changes that affect project infrastructure have a broader blast radius than a
typical feature or bug fix. For this class of change — including, but not
limited to:

- GitHub Actions workflows, CI configuration, or release workflows
- Adding, removing, or bumping dependencies (runtime or development)
- Changes to `pyproject.toml`, `environment.yml`, `requirements*.txt`, or
  `.pre-commit-config.yaml`
- Security-sensitive areas (credentials, file I/O path handling, subprocess
  invocation)

we ask that contributors **open an issue first** to discuss the motivation and
scope before submitting a PR, and explicitly note whether AI tools were used in
the proposed change. AI tools are not a reliable guide to the security,
licensing, or maintenance implications of adding a new dependency, and xradar —
like other scientific open-source projects — is a potential target for
supply-chain-style contributions. Maintainers may ask that AI not be used for
these changes, or that an issue-first discussion happen before any code is
generated.

## Copyright and Licensing

xradar is distributed under the MIT license. By submitting a contribution, you
confirm that you have the right to submit it under that license, whether or not
AI tools were involved.

It is not yet settled whether AI-generated code can be copyrighted, or whether
it can be a derivative of the code the model was trained on. AI tools can
reproduce existing code, including code under incompatible (e.g. copyleft)
licenses, without saying so. Therefore:

- Do not submit AI-generated code that you recognize, or suspect, as copied or
  closely adapted from another project. If you deliberately port code from
  another project, say so in the PR, credit the source, and make sure its
  license is compatible with MIT, exactly as you would for hand-written code.
- Be especially careful with larger, self-contained pieces of generated code
  (complete algorithms, format decoders, lookup tables), which are the most
  likely to be reproduced from training data.
- Maintainers may ask you to rewrite, or may decline, AI-generated code whose
  origin is unclear, and may remove such code later if a problem comes to light.

## Documentation

The same core principles apply to both code and documentation. You must review
the result for accuracy and are ultimately responsible for all changes made.
xradar has domain-specific semantics — radar data conventions, CfRadial2/FM301
terminology, format-specific ICD quirks — that AI tools frequently get wrong.
Do not submit documentation that you haven't carefully read and verified
against authoritative references (e.g. the WMO Manual on Codes, format ICDs,
or the relevant vendor documentation).

## Disclosing AI Usage

When you use AI tools to help with a contribution, we **recommend** noting this
in the PR description, including the tool or model name and version where it is
known (for example: "Claude Opus 4.7", "Cursor with GPT-5.1"). This is not
required, but it helps reviewers calibrate their attention and helps the
community develop shared intuition about where AI tools work well and where
they struggle on radar-data work.

## Revisiting This Policy

AI tools, community practice, and the legal situation around AI-generated code
are all changing quickly. Maintainers will revisit this policy regularly, at
least once per release cycle, and update it as needed. Suggestions are welcome
via an issue.
