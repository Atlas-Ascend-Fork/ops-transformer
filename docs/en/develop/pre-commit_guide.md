# pre-commit Configuration Guide

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-12T12:14:53.135Z pushedAt=2026-11-18T02:51:06.812Z nl=2 -->

## Quick Start

```bash
# 1. Install pre-commit
pip3 install pre-commit

# 2. Install Git Hooks (git commit then automatically triggers checks)
cd /path/to/ops-transformer
pre-commit install        # Cancel: pre-commit uninstall

# 3. Configure the git pc alias (run checks on commits in the specified scope, run once per environment)
git config --global alias.pc '!f() { git diff --name-only "$@" >/dev/null || return 1; git diff --name-only -z "$@" | xargs -0 -r pre-commit run --files; }; f'
git pc HEAD~x        # Check the recent x commits
```

After the preceding configuration is complete, checks run automatically on `git commit` with no manual intervention. See the following sections for full details.

## 1. Overview

pre-commit is a Git Hooks framework that automatically runs code checks and formatting on `git commit`, intercepting specification issues in advance to avoid remote CI gate failures.

| Hook | Function |
|------|------|
| **pre-commit-hooks** | Trailing whitespace, end-of-file line break, YAML/JSON validity, merge conflict markers, private key detection (large files are not intercepted) |
| **clang-format** | C/C++/asc code formatting, following `.clang-format` |
| **ruff-check / ruff-format** | Python static check (auto-fix) + formatting |
| **codespell** | Spell check (terms such as CANN/ascend are whitelisted) |
| **Duplicate Header Check** | Intercepts duplicate basename of added/renamed header files |
| **OAT Check** | Open source compliance check (license header, binary/archive file interception) |

## 2. Environment Requirements

- **Git**: 2.0+, **Python**: 3.9+, **pre-commit**: 4.0+

> clang-format / ruff / codespell are automatically downloaded on the first run of pre-commit (versions are in `.pre-commit-config.yaml`); OAT automatically runs `pip install oat-py` on the first run. No manual installation is required.

## 3. Installation and Configuration

```bash
# 1. Install pre-commit
pip3 install pre-commit

# 2. Install Git Hooks
cd /path/to/ops-transformer
pre-commit install        # Cancel: pre-commit uninstall

# 3. Configure the git pc alias (run checks on commits in the specified scope, run once per environment)
git config --global alias.pc '!f() { git diff --name-only "$@" >/dev/null || return 1; git diff --name-only -z "$@" | xargs -0 -r pre-commit run --files; }; f'
git pc HEAD~x        # Check the recent x commits
```

## 4. Daily Use

```bash
# Automatic check on commit
git add <file list> && git commit -m "msg"

# Manual run
pre-commit run                        # Staging area
pre-commit run --all-files            # All files
pre-commit run clang-format           # A single hook

# Check specified files
pre-commit run --files src/foo.py src/bar.cpp

# Check a specified directory (pre-commit does not support directory arguments; expand to a file list)
pre-commit run --files $(git ls-files examples)

# Skip (emergency)
git commit --no-verify -m "msg"
```

## 5. Failure Troubleshooting

Search for `Failed` in the output to locate the failed item, check the output below that hook to handle it, or paste it to AI for fix suggestions.

Reports of intercepting hooks are written to disk (not tracked by git) and can be reviewed at any time:

- `codespell_reports/codespell.log` (spelling hit checklist)
- `oat_reports/result.txt` (OAT remote hook, compliance violation details)

| Error                  | Cause               | Handling                                          |
|-----------------------|--------------------|---------------------------------------------------|
| clang-format Failed   | The code has specification issues | The hook automatically fixes files; run `git add` again and then commit |
| OAT Compliance Failed | Missing copyright notice, etc.       | Search `OAT Scan Result Summary` to locate the file; the copyright notice cannot be auto-corrected and must be manually completed by referring to the file header in the repository |

## 6. Re-check Historical Commits (After Skipping with --no-verify)

Re-run checks on the changed files of commits in the specified scope, equivalent to the CI `codecheck_precommit` gate:

```bash
git pc HEAD~x        # The recent x commits
git pc <commit>^     # A single commit
```

> Some hooks automatically fix files; after fixing, run `git add` again and append a commit.

## 7. Check Item Description

1. **pre-commit-hooks** (v4.6.0): trailing-whitespace, end-of-file-fixer, check-yaml/json, check-merge-conflict, detect-private-key (check-added-large-files is not enabled)
2. **clang-format** (v18.1.8): follows the project `.clang-format` (Google style, 4-space indentation, column width limit 120, enum left brace on the same line as enum, constructor initializer list line-broken one per line, function definition brace on a new line, pointer left-aligned `int* ptr`; `build/`, `tests/third_party/`, `*_def.cpp` are not formatted)
3. **ruff** (v0.14.14): ruff-check (`--fix` auto-fix) + ruff-format
4. **codespell** (v2.4.1): spell check, terms such as CANN, ascend, EnQue are whitelisted
5. **OAT**: remote compliance hook (HiAgent/compliance, rev v1.0.6, the business repository does not maintain the script), based on oat-py to check license headers (YAML/CSV are exempt) and prohibit binary and archive files
6. **Duplicate Header Basename Check** (local script): intercepts header files with duplicate basename after addition/rename
7. **Compile Scope Whitelist Check** (local script): triggered only when `tests/compile_scope_whitelist.yaml` changes, validates the compile scope whitelist

## 8. FAQ

**Q1: Is the OAT check slow on the first commit?**

The first run requires `pip install oat-py`, which is normal; subsequent runs are fast.

**Q2: Manually format all C/C++?**

```bash
bash scripts/format_cpp.sh            # The entire repository
bash scripts/format_cpp.sh examples   # A specified directory
```

Automatically excludes `build/`, `build_out/`, `third_party/`, `.git/`.

**Q3: How to make it take effect globally so that newly cloned repositories inherit it automatically?**

```bash
mkdir -p ~/.git-templates
pre-commit init-templatedir ~/.git-templates
git config --global init.templatedir ~/.git-templates

# Existing repositories need to re-copy the template
cd /path/to/ops-transformer && git init
```

> The hook is only a launcher; the actual check still reads the current repository's `.pre-commit-config.yaml`; repositories without this file are automatically skipped and do not affect each other.

Cancel: `git config --global --unset init.templatedir && rm -rf ~/.git-templates`

**Q4: When is a local clang-format installation needed?**

Only when manually batch formatting (`bash scripts/format_cpp.sh`), 18.x is recommended; the pre-commit triggered by daily `git commit` does not depend on this tool.

```bash
sudo apt install clang-format
```

## Related Documentation

- [pre-commit documentation](https://pre-commit.com/)
- [clang-format configuration](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT](https://atomgit.com/openharmony-sig/tools_oat)
- [Repository pre-commit integration guide](https://atomgit.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
