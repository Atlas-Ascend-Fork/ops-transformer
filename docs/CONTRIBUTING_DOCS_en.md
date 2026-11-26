# Documentation Contribution Guide

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:14:08.019Z pushedAt=2026-11-18T08:25:31.494Z nl=2 -->

Welcome to contribute to this project's documentation. High-quality documentation is crucial to project success. This guide helps you efficiently submit documentation that complies with the specification.

## Contribution Scope

We welcome any contribution that improves documentation quality, including but not limited to:

- Correction and improvement: Fix typos, grammar errors, incorrect code examples, outdated information, or broken links.

- Clarification and optimization: Make descriptions clearer and easier to understand, optimize sentence structure, and supplement background knowledge.

- Content supplement: Add usage examples, API documentation, FAQ, best practices, or warning notes for existing functions.

- New content creation: Write new chapters or tutorials for new functions, such as operator README, API introduction documents, etc. If you have questions, create an Issue for discussion first.

- Localization: Help us translate or proofread documentation in other languages.

- Style and navigation: Improve the layout, readability, and navigation structure of the documentation website.

## Contribution Process

1. **Prepare**

    - Determine the task: If you find documentation issues, create new Issues. Use the label `[Documentation|Documentation Feedback]` and provide a detailed description. Based on the existing Issues list, identify the documentation issue to resolve.
    - Claim the task: Comment `/assign @yourself` under the corresponding Issue to indicate that you will handle it and avoid duplicate work.

2. **Modify Documentation**

    - Select a branch: Download the source code from the master branch or another Tag branch to your local machine.
    - Follow the format:
      - This project recommends using **Markdown format**.
      - Follow the project's existing writing style.
      - Place static resources such as images in the corresponding directory. For example, images are generally placed in the `figures` folder under the docs directory. Adjust as needed in special cases.
    - Add and delete carefully: When modifying content, try to keep the original line width and line break conventions.

3. **Submit Changes**

    - Atomic commits: Each commit should focus on one independent change. For example, "Fix typos in the xx guide" and "Update the sample code in the API reference" should be two separate commits.

    - Write clear commit messages:

      ```text
      Brief description (no more than 50 characters)

      If necessary, provide a more detailed description here. Explain the reason and content of the change, not what was specifically changed (the code itself shows that).
      Related Issue: #123
      ```

4. **Initiate a Pull Request**

    - Target branch: Merge the PR into the project's target branch.
    - Title and description:
      - PR title: Clearly summarize the change, for example: `[Docs] Fix the configuration example in the quick start guide`.
      - PR description: Describe your changes and motivation in detail, and associate the corresponding Issue (use Closes #123 or Fixes #456).
    - Preview check: Check the rendered documentation locally or online in advance to ensure it renders as expected.
    - Wait for review: Maintainers will review and may provide modification suggestions. Follow up on the discussion in a timely manner.

## Writing Specification

Before writing project documentation, developers must read the following specifications. If you have any questions, feel free to raise suggestions at any time.

- Prerequisites: First learn the unified writing specification provided by the CANN organization. For details, see [CANN Documentation Writing Specification](https://atomgit.com/cann/community/blob/master/contributor/docs/document_writing_specs.md).

  - Documentation content requirements: Introduce the mandatory and optional documentation deliverables in the project.
  - Directory structure specification: Introduce the principles of directory division, such as Chinese and English management.
  - Content element specification: Introduce the rules for different writing elements, such as file naming, titles, fonts, images, code blocks, links, etc.

- Notes:

  In addition to the above writing rules, note the following items:

  - Tone: Use a friendly, professional, and neutral tone. Target beginners and avoid unnecessary jargon.
  - Terminology: Keep terminology consistent (for example, use "click" consistently). Refer to the project glossary (if any).
  - Code examples:
    - Ensure all code examples are runnable and tested.
    - Provide sufficient context and explanation.
    - Note the environment or prerequisites required to run the code.
  - Punctuation and format:
    - When mixing Chinese and English, use full-width punctuation, and punctuation must comply with the Chinese/English context.
    - Use appropriate heading levels (#, ##, ###).
    - Use lists and tables to organize complex information.
  - Links: Use descriptive link text, avoid "click here", and ensure link resources are authentic and reliable.
  - Images:
    - Common format: PNG is recommended. Try to keep the style consistent with existing images.
    - Resolution and clarity: Images must be clear and appropriately sized. Avoid blurriness or excessive compression.
    - File size: A single image should not exceed 10 MB.
  - Copyright: Ensure compliance for all referenced images, literature, and other resources.

## Get Help

If you have any questions during the contribution process:

1. Check existing documentation: If you have questions about templates or specifications, first check the project's existing guides, API documentation, or README.
2. Start a discussion: You can create a new Issue or leave a comment directly in the relevant Issue or PR.

## Documentation Template

The key documents involved in operator deliverables mainly include the following. For specific writing formats and content requirements, refer to the templates.

- [Operator README Document Template](https://atomgit.com/cann/ops-transformer/wiki/operator-readme-document-template)
- [aclnn API Document Template](https://atomgit.com/cann/ops-transformer/wiki/aclnn-api-document-template)
