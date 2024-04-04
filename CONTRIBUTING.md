# Contributing to Jupyter Scatter

:wave: Welcome and thank you for considering contributing to Jupyter Scatter.

In this guide you will get an overview of the contribution workflow from opening an issue, creating a pull request (PR), reviewing, and merging the PR.

## Getting Started

To get an overview of Jupyter Scatter, take a look at the [_Get Started_ guide](https://jupyter-scatter.dev/get-started) and check out our [tutorial](https://github.com/flekschas/jupyter-scatter-tutorial).

## Help! I am lost... :sob:

Head over to our [discussion board](https://github.com/flekschas/jupyter-scatter/discussions) and post any question you have! We're happy to help.

## Reporting Issues :bug: or Feature Requests :bulb:

If you discover a bug or have an idea for a new feature, please [submit an issue](https://github.com/flekschas/jupyter-scatter/issues/new/choose). Make sure the issue is following either the [bug report](/.github/ISSUE_TEMPLATE/bug.md) or [feature request](/.github/ISSUE_TEMPLATE/enhancement.md) template.

For questions on how to use Jupyter Scatter, please use the [discussion board](https://github.com/flekschas/jupyter-scatter/discussions).

## Contribute Code :nail_care:

Do you have a concrete plan how to improve Jupyter Scatter and have discussed your ideas as part of an [issue](#reporting-issues-bug-or-feature-requests-bulb)? Fantastic! :star_struck: Submit a PR! We welcome your contribution. If you're wondering what a PR is, please check out [GitHub's how to](https://help.github.com/articles/using-pull-requests/).

A maintainer will review your pull request. Once the PR is approved and passes continuous integration checks, we will merge it and add it to the next scheduled release.

### Pull Request Checklist :white_check_mark:

- Ensure that your PR answers the relevant questions in the [PR template](/.github/PULL_REQUEST_TEMPLATE.md).
- Ensure that your PR successfully passes all continuous integration checks.
- Optionally, if your PR introduces a new feature, it'd be great to have a code snippet and an image/video showcasing the new functionality.

### Reviewing Pull Requests :eyes:

When reviewing a PR, first verify whether the PR addresses a valid concern by checking the associated [issue](#reporting-issues-bug-or-feature-requests-bulb) and answers the relevant questions in the [PR template](/.github/PULL_REQUEST_TEMPLATE.md).

**:white_check_mark: Examples of Valid Concerns:**

- PR addressing a documented and verified bug
- PR improving the performance or documentation
- PR introducing a new feature that is well described and fits the scope of Jupyter Scatter 

**:question: Examples of Questionable Concerns:**

- PR addressing an undocumented issue or where it's unclear whether the issue is a bug or an unsupported feature
- PR improving the performance of an edge-case by introducing a lot of new code
- PR introducing a new feature that is too niche, is too application-specific, does not scalable, or does not generalize

For questionable PRs with concerns, try to clarify details with the PR creator and potentially get other maintainers involved to come to a conclusion whether the PR should eventually be accepted or not.

Second, for acceptable PRs, verify that the code changes are valid, concise, pythonic, performant, and follow our conventions.

### Merging Pull Requests :twisted_rightwards_arrows:

Once a PR passes all continuous integration checks and has been reviewed, merge the beauty. If the PR addresses a bug, immediately release a patch. If the PR introduces a new feature, check with the other maintainers and potentially release a 
minor or major version. Major version bumps should only occur for breaking changes!
