#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared incremental msprof-to-Excel merge support for SMLA variants."""

import logging
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


LOGGER = logging.getLogger(__name__)
OP_NAME_COLUMN = "Op Name"
TASK_DURATION_COLUMN = "Task Duration(us)"
MODEL_ID_COLUMNS = ("Model ID", "Model Id")
# msprof uses UINT32_MAX for compile/step records outside a graph execution.
STEP_MODEL_ID = (1 << 32) - 1
INVALID_RESULTS = {"npu error", "exception"}
TESTCASE_NAME_COLUMN = "testcase_name"
TESTCASE_NAME_ALIASES = (
    TESTCASE_NAME_COLUMN,
    "Testcase_Name",
    "case_name",
)


@dataclass(frozen=True)
class OperatorPerfConfig:
    """Operator-specific names used while reading an msprof op_summary."""

    operator_label: str
    operator_token: str
    metadata_token: str
    duration_column: str = "qsmla_duration"
    excluded_tokens: tuple = ()
    case_columns: tuple = (TESTCASE_NAME_COLUMN,)
    # Opt into SMLA's graph-record selection and persist the selected model ID.
    model_id_column: str = ""
    perf_diff_column: str = "qsmla_perf_diff"
    legacy_column_aliases: tuple = ()
    case_name_normaliser: object = None

    @property
    def compare_columns(self):
        return ["metadata_perf_diff", self.perf_diff_column, "perf_result"]

    @property
    def perf_columns(self):
        columns = [
            "profile_folder",
            "profile_run",
            "metadata_duration",
            self.duration_column,
        ]
        if self.model_id_column:
            columns.append(self.model_id_column)
        return columns


SMLA_PERF_CONFIG = OperatorPerfConfig(
    operator_label="SparseFlashMla",
    operator_token="sparseflashmla",
    metadata_token="sparseflashmlametadata",
    excluded_tokens=("quantsparseflashmla",),
    duration_column="smla_duration",
    model_id_column="smla_model_id",
)
SMLA_PERF_COLUMNS = [
    "profile_folder",
    "profile_run",
    SMLA_PERF_CONFIG.model_id_column,
    "metadata_duration",
    SMLA_PERF_CONFIG.duration_column,
]
QSMLA_PERF_CONFIG = OperatorPerfConfig(
    operator_label="QuantSparseFlashMla",
    operator_token="quantsparseflashmla",
    metadata_token="quantsparseflashmlametadata",
    excluded_tokens=("mixedquantsparseflashmla",),
    model_id_column="qsmla_model_id",
)


def mqsmla_perf_config(case_name_normaliser):
    return OperatorPerfConfig(
        operator_label="MixedQuantSparseFlashMla",
        operator_token="mixedquantsparseflashmla",
        metadata_token="mixedquantsparseflashmlametadata",
        duration_column="mqsmla_duration",
        model_id_column="mqsmla_model_id",
        perf_diff_column="mqsmla_perf_diff",
        legacy_column_aliases=(
            ("qsmla_duration", "mqsmla_duration"),
            ("qsmla_perf_diff", "mqsmla_perf_diff"),
        ),
        case_name_normaliser=case_name_normaliser,
    )


@dataclass(frozen=True)
class PerfSample:
    """One main-operator invocation and its matching metadata invocation."""

    profile_folder: str
    profile_run: int
    metadata_duration: object = None
    operator_duration: object = None
    operator_model_id: object = None


def _normalise_op_name(op_name):
    return re.sub(r"[^a-z0-9]", "", str(op_name).lower())


def _classify_op(op_name, config):
    normalised = _normalise_op_name(op_name)
    if any(token in normalised for token in config.excluded_tokens):
        return None
    if config.metadata_token in normalised:
        return "metadata"
    if config.operator_token in normalised:
        return "operator"
    return None


def _normalise_model_id(model_id):
    if model_id is None or pd.isna(model_id):
        return None
    try:
        numeric_id = float(model_id)
        if numeric_id.is_integer():
            return int(numeric_id)
    except (TypeError, ValueError, OverflowError):
        pass
    return str(model_id).strip()


def _get_model_id(row):
    for column in MODEL_ID_COLUMNS:
        if column in row.index:
            return _normalise_model_id(row[column])
    return None


def _is_graph_model_id(model_id):
    return isinstance(model_id, int) and 0 <= model_id < STEP_MODEL_ID


def _select_operator_records(records, profile_folder, config):
    if not config.model_id_column:
        return records
    graph_records = [record for record in records if _is_graph_model_id(record[1])]
    if not graph_records:
        return records

    ignored_count = len(records) - len(graph_records)
    if ignored_count:
        LOGGER.info(
            "%s: use %d graph-model %s record(s), ignore %d compile/step record(s)",
            Path(profile_folder).name,
            len(graph_records),
            config.operator_label,
            ignored_count,
        )
    return graph_records


def _profile_sort_key(profile_folder):
    try:
        modified_time = profile_folder.stat().st_mtime_ns
    except OSError:
        modified_time = 0
    return modified_time, profile_folder.name


def find_profile_folders(profile_root):
    """Return all msprof directories in collection order."""
    root = Path(profile_root)
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("PROF")
        ),
        key=_profile_sort_key,
    )


def _read_op_summary_files(profile_folder):
    profiler_output = profile_folder / "mindstudio_profiler_output"
    if not profiler_output.is_dir():
        LOGGER.warning(
            "Skip %s: mindstudio_profiler_output does not exist", profile_folder
        )
        return []

    summary_files = sorted(profiler_output.glob("op_summary*.csv"))
    if not summary_files:
        LOGGER.warning("Skip %s: op_summary CSV does not exist", profile_folder)
        return []

    summaries = []
    for summary_file in summary_files:
        try:
            summary = pd.read_csv(summary_file)
        except Exception as error:  # keep processing other profiler outputs
            LOGGER.warning("Failed to read %s: %s", summary_file, error)
            continue
        missing_columns = {
            OP_NAME_COLUMN,
            TASK_DURATION_COLUMN,
        } - set(summary.columns)
        if missing_columns:
            LOGGER.warning(
                "Skip %s: missing columns %s",
                summary_file,
                sorted(missing_columns),
            )
            continue
        summaries.append(summary)
    return summaries


def extract_profile_samples(profile_folder, config):
    """Extract all target-operator measurements from one PROF directory."""
    metadata_durations = []
    operator_records = []
    latest_metadata_duration = None

    for summary in _read_op_summary_files(Path(profile_folder)):
        for _, row in summary.iterrows():
            op_type = _classify_op(row[OP_NAME_COLUMN], config)
            if op_type == "metadata":
                metadata_durations.append(row[TASK_DURATION_COLUMN])
                latest_metadata_duration = row[TASK_DURATION_COLUMN]
            elif op_type == "operator":
                operator_records.append(
                    (
                        row[TASK_DURATION_COLUMN],
                        _get_model_id(row),
                        latest_metadata_duration,
                    )
                )

    operator_records = _select_operator_records(
        operator_records, profile_folder, config
    )

    pair_metadata_records = config.model_id_column and bool(operator_records)
    # Compilation and warmup can execute metadata more than once. Pair the
    # latest preceding metadata with each retained main execution; metadata-only
    # records must not create extra testcase rows.
    sample_count = (
        len(operator_records)
        if pair_metadata_records
        else max(len(metadata_durations), len(operator_records))
    )
    samples = []
    for index in range(sample_count):
        operator_duration, operator_model_id, paired_metadata_duration = (
            operator_records[index]
            if index < len(operator_records)
            else (None, None, None)
        )
        samples.append(
            PerfSample(
                profile_folder=Path(profile_folder).name,
                profile_run=index + 1,
                metadata_duration=(
                    paired_metadata_duration
                    if pair_metadata_records
                    else (
                        metadata_durations[index]
                        if index < len(metadata_durations)
                        else None
                    )
                ),
                operator_duration=operator_duration,
                operator_model_id=operator_model_id,
            )
        )
    return samples


def collect_profile_samples(profile_root, config):
    """Extract target-operator measurements from every PROF directory."""
    samples = []
    for profile_folder in find_profile_folders(profile_root):
        folder_samples = extract_profile_samples(profile_folder, config)
        LOGGER.info(
            "%s: collected %d %s run(s)",
            profile_folder.name,
            len(folder_samples),
            config.operator_label,
        )
        samples.extend(folder_samples)
    return samples


def _valid_result_indices(result_frame):
    if "result" not in result_frame.columns:
        return list(result_frame.index)
    statuses = result_frame["result"].fillna("").astype(str).str.strip().str.lower()
    return [
        index
        for index in result_frame.index
        if statuses.at[index] not in INVALID_RESULTS
    ]


def _same_value(left, right):
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if left_missing or right_missing:
        return bool(left_missing and right_missing)
    return left == right


def normalise_testcase_name_column(frame, required=False):
    """Expose the source testcase name unchanged through one canonical column."""
    source_columns = [
        column for column in TESTCASE_NAME_ALIASES if column in frame.columns
    ]
    if not source_columns:
        if required:
            raise ValueError(
                "Test result must contain one testcase-name column: "
                f"{', '.join(TESTCASE_NAME_ALIASES)}"
            )
        return frame.copy()

    normalised = frame.copy()
    testcase_names = normalised[source_columns[0]].copy()
    for source_column in source_columns[1:]:
        source_names = normalised[source_column]
        conflicts = [
            index
            for index in normalised.index
            if not pd.isna(testcase_names.at[index])
            and not pd.isna(source_names.at[index])
            and not _same_value(
                testcase_names.at[index],
                source_names.at[index],
            )
        ]
        if conflicts:
            raise ValueError(
                "Conflicting testcase names found in columns "
                f"{source_columns[0]} and {source_column}, row {conflicts[0]}"
            )
        testcase_names = testcase_names.where(testcase_names.notna(), source_names)

    normalised = normalised.drop(columns=source_columns)
    normalised.insert(0, TESTCASE_NAME_COLUMN, testcase_names)
    return normalised


def replace_testcase_names_with_filenames(
    frame,
    testcase_paths,
    strip_pt_suffix=False,
    case_name_normaliser=None,
):
    """Backfill historical logical case names with unique .pt basenames."""
    if TESTCASE_NAME_COLUMN not in frame.columns:
        return frame.copy()

    testcase_filenames = set()
    for testcase_path in testcase_paths or ():
        if not testcase_path:
            continue
        path = Path(testcase_path)
        if path.is_file() and path.suffix.lower() == ".pt":
            testcase_filenames.add(path.name)
        elif path.is_dir():
            testcase_filenames.update(
                child.name
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() == ".pt"
            )

    normalised_filenames = (
        {case_name_normaliser(filename) for filename in testcase_filenames}
        if case_name_normaliser is not None
        else set()
    )
    resolved = frame.copy()
    for index, testcase_name in resolved[TESTCASE_NAME_COLUMN].items():
        if pd.isna(testcase_name):
            continue
        current_name = Path(str(testcase_name)).name
        if case_name_normaliser is not None:
            current_name = case_name_normaliser(current_name)
            matches = sorted(
                name
                for name in normalised_filenames
                if name == current_name or name.endswith(f"_{current_name}")
            )
            if current_name in normalised_filenames:
                resolved.at[index, TESTCASE_NAME_COLUMN] = current_name
            elif len(matches) == 1:
                resolved.at[index, TESTCASE_NAME_COLUMN] = matches[0]
            else:
                resolved.at[index, TESTCASE_NAME_COLUMN] = current_name
                if len(matches) > 1:
                    LOGGER.warning(
                        "Cannot expand testcase_name %r: it matches multiple .pt files: %s",
                        current_name,
                        matches,
                    )
            continue
        if current_name.lower().endswith(".pt"):
            resolved.at[index, TESTCASE_NAME_COLUMN] = (
                Path(current_name).stem if strip_pt_suffix else current_name
            )
            continue

        expected_suffix = f"_{current_name}.pt"
        matches = sorted(
            filename
            for filename in testcase_filenames
            if filename.endswith(expected_suffix)
        )
        if len(matches) == 1:
            resolved.at[index, TESTCASE_NAME_COLUMN] = (
                Path(matches[0]).stem if strip_pt_suffix else matches[0]
            )
        elif len(matches) > 1:
            LOGGER.warning(
                "Cannot expand testcase_name %r: it matches multiple .pt files: %s",
                current_name,
                matches,
            )
    return resolved


def _prepare_perf_frame(frame, config, testcase_paths, required=False):
    """Normalise historical result, performance and baseline schemas alike."""
    frame = frame.copy()
    for legacy_column, column in config.legacy_column_aliases:
        if legacy_column not in frame.columns:
            continue
        if column in frame.columns:
            conflicts = [
                index
                for index in frame.index
                if not pd.isna(frame.at[index, column])
                and not pd.isna(frame.at[index, legacy_column])
                and not _same_value(
                    frame.at[index, column], frame.at[index, legacy_column]
                )
            ]
            if conflicts:
                raise ValueError(
                    f"Conflicting performance columns {legacy_column} and {column}, "
                    f"row {conflicts[0]}"
                )
            frame[column] = frame[column].where(
                frame[column].notna(), frame[legacy_column]
            )
            frame = frame.drop(columns=legacy_column)
        else:
            frame = frame.rename(columns={legacy_column: column})
    if config.case_name_normaliser is not None:
        for column in TESTCASE_NAME_ALIASES:
            if column in frame.columns:
                frame[column] = frame[column].map(config.case_name_normaliser)
    frame = normalise_testcase_name_column(frame, required=required)
    return replace_testcase_names_with_filenames(
        frame,
        testcase_paths,
        strip_pt_suffix=True,
        case_name_normaliser=config.case_name_normaliser,
    )


def _same_result_row(result_frame, existing_frame, index):
    return all(
        column in existing_frame.columns
        and _same_value(
            result_frame.at[index, column], existing_frame.at[index, column]
        )
        for column in result_frame.columns
    )


def _sample_key(sample):
    values = (
        sample.profile_folder,
        sample.profile_run,
        sample.metadata_duration,
        sample.operator_duration,
    )
    return tuple(None if pd.isna(value) else value for value in values)


def _row_sample_key(row, config):
    columns = [
        "profile_folder",
        "profile_run",
        "metadata_duration",
        config.duration_column,
    ]
    return tuple(
        None if pd.isna(row.get(column)) else row.get(column) for column in columns
    )


def _assign_sample(frame, index, sample, config):
    frame.at[index, "profile_folder"] = sample.profile_folder
    frame.at[index, "profile_run"] = sample.profile_run
    frame.at[index, "metadata_duration"] = sample.metadata_duration
    frame.at[index, config.duration_column] = sample.operator_duration
    if config.model_id_column:
        frame.at[index, config.model_id_column] = sample.operator_model_id


def merge_results_and_perf(result_frame, samples, config, existing_frame=None):
    """Map new samples to pytest rows while preserving all earlier runs."""
    merged = result_frame.copy()
    managed_columns = config.perf_columns + config.compare_columns
    for column in managed_columns:
        merged[column] = pd.NA

    if existing_frame is not None:
        shared_row_count = min(len(merged), len(existing_frame))
        for index in range(shared_row_count):
            if not _same_result_row(result_frame, existing_frame, index):
                continue
            for column in managed_columns:
                if column in existing_frame.columns:
                    merged.at[index, column] = existing_frame.at[index, column]

    valid_indices = _valid_result_indices(merged)
    empty_indices = [
        index
        for index in valid_indices
        if pd.isna(merged.at[index, "metadata_duration"])
        and pd.isna(merged.at[index, config.duration_column])
    ]
    existing_keys = {
        _row_sample_key(row, config)
        for _, row in merged.iterrows()
        if not (
            pd.isna(row["metadata_duration"]) and pd.isna(row[config.duration_column])
        )
    }
    new_samples = [
        sample for sample in samples if _sample_key(sample) not in existing_keys
    ]
    # msprof may reuse and overwrite the same PROF directory. Newly appended
    # pytest rows prove that the current measurements belong to a new run even
    # when the folder name and durations happen to be identical.
    if empty_indices and len(new_samples) < len(empty_indices):
        new_samples = samples[-len(empty_indices) :]

    for result_index, sample in zip(empty_indices, new_samples):
        _assign_sample(merged, result_index, sample, config)

    unmatched_samples = new_samples[len(empty_indices) :]
    if unmatched_samples:
        LOGGER.warning(
            "New profiler runs (%d) exceed empty pytest rows (%d); "
            "retaining %d run(s) as performance-only rows",
            len(new_samples),
            len(empty_indices),
            len(unmatched_samples),
        )
        extra_rows = []
        for sample in unmatched_samples:
            row = {column: pd.NA for column in merged.columns}
            extra_rows.append(row)
        extra_frame = pd.DataFrame(extra_rows)
        start_index = len(merged)
        merged = pd.concat([merged, extra_frame], ignore_index=True)
        for offset, sample in enumerate(unmatched_samples):
            _assign_sample(merged, start_index + offset, sample, config)

    return merged


def _matching_golden_row(result_row, result_index, golden_frame, config):
    for case_column in config.case_columns:
        if (
            case_column not in result_row.index
            or case_column not in golden_frame.columns
        ):
            continue
        case_value = result_row[case_column]
        if pd.isna(case_value):
            continue
        matches = golden_frame[golden_frame[case_column] == case_value]
        if not matches.empty:
            return matches.iloc[0]
    if result_index < len(golden_frame):
        return golden_frame.iloc[result_index]
    return None


def apply_perf_comparison(merged, golden_frame, config, perf_threshold=10):
    """Populate the legacy performance comparison columns when requested."""
    failed_cases = []
    for index, row in merged.iterrows():
        current_operator = pd.to_numeric(
            pd.Series([row.get(config.duration_column)]), errors="coerce"
        ).iloc[0]
        if pd.isna(current_operator):
            continue
        golden_row = _matching_golden_row(row, index, golden_frame, config)
        if golden_row is None:
            continue

        golden_operator = pd.to_numeric(
            pd.Series([golden_row.get(config.duration_column)]), errors="coerce"
        ).iloc[0]
        current_metadata = pd.to_numeric(
            pd.Series([row.get("metadata_duration")]), errors="coerce"
        ).iloc[0]
        golden_metadata = pd.to_numeric(
            pd.Series([golden_row.get("metadata_duration")]), errors="coerce"
        ).iloc[0]

        if not pd.isna(current_metadata) and not pd.isna(golden_metadata):
            merged.at[index, "metadata_perf_diff"] = current_metadata - golden_metadata
        if pd.isna(golden_operator):
            continue
        operator_diff = current_operator - golden_operator
        merged.at[index, config.perf_diff_column] = operator_diff
        merged.at[index, "perf_result"] = (
            "Failed" if abs(operator_diff) > perf_threshold else "Pass"
        )
        if abs(operator_diff) > perf_threshold:
            case_name = next(
                (
                    row.get(column)
                    for column in config.case_columns
                    if column in row.index and not pd.isna(row.get(column))
                ),
                f"row_{index}",
            )
            failed_cases.append(case_name)
    return failed_cases


def _recorded_runs(table):
    return {
        (row.profile_folder, row.profile_run): index
        for index, row in table[["profile_folder", "profile_run"]].iterrows()
        if pd.notna(row.profile_folder) and pd.notna(row.profile_run)
    }


def _fill_unmeasured_rows(table, samples, config):
    recorded = _recorded_runs(table)
    unseen = [s for s in samples if (s.profile_folder, s.profile_run) not in recorded]
    if not unseen:
        return
    blanks = (
        table[["profile_folder", "metadata_duration", config.duration_column]]
        .isna()
        .all(axis=1)
    )
    indices = [i for i in _valid_result_indices(table) if blanks[i]]
    if len(unseen) != len(indices):
        raise ValueError(
            "Ambiguous profile rows; use the capture's result sheet with --merge_into"
        )
    for index, sample in zip(indices, unseen):
        _assign_sample(table, index, sample, config)


def _merge_captured_cases(table, capture, samples, config, managed_columns):
    indices = _valid_result_indices(capture)
    if len(samples) != len(indices):
        raise ValueError(
            f"Capture contains {len(indices)} completed case rows but "
            f"{len(samples)} graph measurements; cannot associate them safely"
        )
    if capture[TESTCASE_NAME_COLUMN].isna().any():
        raise ValueError("Missing case name in the captured result sheet")
    for index, sample in zip(indices, samples):
        _assign_sample(capture, index, sample, config)

    # Select the latest accuracy row for each case, preferring its existing
    # measured row so another performance run refreshes that same row.
    targets = dict(zip(table[TESTCASE_NAME_COLUMN], table.index))
    recorded = _recorded_runs(table)
    for index in recorded.values():
        targets[table.at[index, TESTCASE_NAME_COLUMN]] = index
    table = table.reindex(columns=table.columns.union(capture.columns, sort=False))
    compared = set()
    for _, row in capture.iterrows():
        name = row[TESTCASE_NAME_COLUMN]
        key = (row["profile_folder"], row["profile_run"])
        if key in recorded and table.at[recorded[key], TESTCASE_NAME_COLUMN] != name:
            raise ValueError(
                f"Profile record {key} is already assigned to a different case"
            )
        if name not in targets:
            target = len(table)
            table.loc[target] = row
            targets[name] = target
        elif pd.isna(row[config.duration_column]):
            continue
        else:
            target = targets[name]
            # Keep accuracy and user columns. A new timing invalidates the
            # previous baseline comparison until the new timing is compared.
            columns = config.perf_columns if key in recorded else managed_columns
            table.loc[target, columns] = row[columns]
        if pd.notna(row[config.duration_column]):
            compared.add(target)
    return table, sorted(compared)


def update_result_workbook(
    config,
    test_result_path,
    profile_root,
    destination,
    testcase_paths,
    managed_columns,
    is_compare=False,
    perf_golden_path=None,
    perf_threshold=10,
    delete_perf_files=False,
):
    """Attach one capture to its case rows, preserving existing accuracy results."""
    source = Path(test_result_path)
    destination = Path(destination)
    in_place = source.resolve() == destination.resolve()
    samples = collect_profile_samples(profile_root, config)
    if not samples or any(pd.isna(s.operator_duration) for s in samples):
        raise ValueError(
            f"No complete {config.operator_label} measurements in {profile_root}"
        )
    if (
        not in_place
        and config.model_id_column
        and any(not _is_graph_model_id(s.operator_model_id) for s in samples)
    ):
        raise ValueError(
            f"No valid graph measurement for some {config.operator_label} cases in {profile_root}"
        )

    with result_workbook_lock(destination):
        source_rows = read_result_workbook(source)
        capture = _prepare_perf_frame(
            source_rows, config, testcase_paths, required=True
        )
        if in_place:
            original, table = source_rows, capture.copy()
        elif destination.is_file():
            original = read_result_workbook(destination)
            table = _prepare_perf_frame(original, config, testcase_paths, required=True)
        else:
            original, table = source_rows, capture.iloc[:0].copy()
        name_column = next(col for col in TESTCASE_NAME_ALIASES if col in original)
        for col in managed_columns:
            table[col] = table[col].astype(object) if col in table else pd.NA
            capture[col] = pd.NA
        if in_place:
            _fill_unmeasured_rows(table, samples, config)
            compared = table.index
        else:
            table, compared = _merge_captured_cases(
                table, capture, samples, config, managed_columns
            )
        if is_compare:
            golden = _prepare_perf_frame(
                pd.read_excel(perf_golden_path), config, testcase_paths
            )
            subset = table.loc[compared].copy()
            apply_perf_comparison(subset, golden, config, perf_threshold)
            table.loc[compared, config.compare_columns] = subset[config.compare_columns]
        front = [col for col in table if col not in managed_columns]
        table = table[front + list(managed_columns)].rename(
            columns={TESTCASE_NAME_COLUMN: name_column}
        )
        write_result_workbook(table, destination)
    if delete_perf_files:
        for folder in find_profile_folders(profile_root):
            shutil.rmtree(folder)
    LOGGER.info("Saved %s results to %s", config.operator_label, destination)
    return destination


@contextmanager
def result_workbook_lock(path):
    """Serialize read-modify-write operations, including parallel pytest workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a stable lock inode across workbook replacements and waiting writers.
    with path.with_name(f".{path.name}.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def read_result_workbook(path):
    """Read Excel cell values without interpreting text IDs or labels as numbers/NA."""
    return pd.read_excel(path, dtype=object, keep_default_na=False, na_values=[""])


def write_result_workbook(frame, path):
    """Publish the complete workbook with a same-directory atomic replacement."""
    path = Path(path)
    with TemporaryDirectory(dir=path.parent, prefix=f".{path.stem}.") as temporary:
        temporary_path = Path(temporary) / path.name
        frame.to_excel(temporary_path, index=False)
        if path.exists():
            shutil.copymode(path, temporary_path)
        os.replace(temporary_path, path)


def append_accuracy_result(path, row_data, config, perf_columns=None):
    """Append one accuracy execution with blank profiling cells in the result sheet."""
    path = Path(path)
    if perf_columns is None:
        perf_columns = config.perf_columns + config.compare_columns
    new_row = _prepare_perf_frame(pd.DataFrame([row_data]), config, ())
    new_row = new_row.drop(columns=perf_columns, errors="ignore")
    with result_workbook_lock(path):
        existing = read_result_workbook(path) if path.is_file() else pd.DataFrame()
        case_column = next(
            (
                name
                for name in (*existing.columns, *row_data)
                if name in TESTCASE_NAME_ALIASES
            ),
            TESTCASE_NAME_COLUMN,
        )
        existing = _prepare_perf_frame(existing, config, ())
        frame = pd.concat([existing, new_row], ignore_index=True, sort=False)
        trailing_columns = ["result", "fulfill_percent", *perf_columns]
        case_columns = [column for column in frame if column not in trailing_columns]
        frame = frame.reindex(columns=case_columns + trailing_columns)
        frame = frame.rename(columns={TESTCASE_NAME_COLUMN: case_column})
        write_result_workbook(frame, path)


def process_all_profiler_data(
    config,
    test_result_path,
    profile_root=".",
    output_path=None,
    is_compare=False,
    perf_golden_path="perf_golden.xlsx",
    perf_threshold=10,
    delete_perf_files=False,
    testcase_paths=(),
    merge_into=None,
):
    """Merge every available profiler run and save the table atomically."""
    result_path = Path(test_result_path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Test result file does not exist: {result_path}")
    if merge_into and output_path:
        raise ValueError("--merge_into and --output_path are mutually exclusive")
    if (
        merge_into
        or not output_path
        or Path(output_path).resolve() == result_path.resolve()
    ):
        return update_result_workbook(
            config,
            result_path,
            profile_root,
            merge_into or result_path,
            testcase_paths,
            config.perf_columns + config.compare_columns,
            is_compare,
            perf_golden_path,
            perf_threshold,
            delete_perf_files,
        )

    result_frame = _prepare_perf_frame(
        pd.read_excel(result_path),
        config,
        testcase_paths,
        required=True,
    )
    profile_folders = find_profile_folders(profile_root)
    if not profile_folders:
        raise FileNotFoundError(f"No PROF directory found under: {profile_root}")

    samples = collect_profile_samples(profile_root, config)
    if not samples:
        raise ValueError(
            f"No {config.operator_label} performance data found in PROF directories"
        )

    destination = Path(output_path)
    existing_frame = (
        _prepare_perf_frame(pd.read_excel(destination), config, testcase_paths)
        if destination.is_file()
        else None
    )
    if existing_frame is not None:
        metadata_columns = {
            TESTCASE_NAME_COLUMN,
            "profile_folder",
            "metadata_duration",
            config.duration_column,
        }
        has_metadata_only_extras = False
        if config.model_id_column and metadata_columns.issubset(existing_frame.columns):
            current_folders = {sample.profile_folder for sample in samples}
            has_metadata_only_extras = (
                existing_frame[TESTCASE_NAME_COLUMN].isna()
                & existing_frame[config.duration_column].isna()
                & existing_frame["metadata_duration"].notna()
                & existing_frame["profile_folder"].isin(current_folders)
            ).any()
        if config.model_id_column and (
            has_metadata_only_extras
            or (
                config.model_id_column not in existing_frame.columns
                and any(
                    _is_graph_model_id(sample.operator_model_id) for sample in samples
                )
            )
        ):
            # Legacy output assigned compile/step data to the testcase and kept
            # graph or repeated metadata measurements as nameless extra rows.
            # Rebuild from the selected main executions, as SMLA does.
            LOGGER.info(
                "Rebuild legacy performance output using main-operator records: %s",
                destination,
            )
            existing_frame = None
    merged = merge_results_and_perf(result_frame, samples, config, existing_frame)

    failed_cases = []
    if is_compare:
        golden_frame = _prepare_perf_frame(
            pd.read_excel(perf_golden_path),
            config,
            testcase_paths,
        )
        failed_cases = apply_perf_comparison(
            merged,
            golden_frame,
            config,
            perf_threshold=perf_threshold,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.tmp.xlsx")
    merged.to_excel(temporary_path, index=False)
    os.replace(temporary_path, destination)

    LOGGER.info(
        "Saved %d %s run(s) from %d folder(s) to %s",
        len(samples),
        config.operator_label,
        len(profile_folders),
        destination,
    )
    if failed_cases:
        LOGGER.warning("Performance comparison failed cases: %s", failed_cases)

    if delete_perf_files:
        for profile_folder in profile_folders:
            shutil.rmtree(profile_folder)
            LOGGER.info("Deleted profiler directory: %s", profile_folder)

    return destination
