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

"""Merge all SparseFlashMla msprof runs into one Excel result file."""

import argparse
import logging
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from perf_data_common import (
    SMLA_PERF_COLUMNS as PERF_COLUMNS,
    SMLA_PERF_CONFIG as PERF_CONFIG,
    _is_graph_model_id,
    extract_profile_samples as extract_operator_samples,
    find_profile_folders,
    normalise_testcase_name_column,
    replace_testcase_names_with_filenames,
    update_result_workbook,
)


LOGGER = logging.getLogger(__name__)
SMLA_MODEL_ID_COLUMN = PERF_CONFIG.model_id_column
INVALID_RESULTS = {"npu error", "exception"}


@dataclass(frozen=True)
class PerfSample:
    """One SparseFlashMla invocation and its matching metadata invocation."""

    profile_folder: str
    profile_run: int
    smla_model_id: object = None
    metadata_duration: object = None
    smla_duration: object = None


def extract_profile_samples(profile_folder):
    """Use the common graph/metadata pairing and keep SMLA's sample schema."""
    return [
        PerfSample(
            profile_folder=sample.profile_folder,
            profile_run=sample.profile_run,
            smla_model_id=sample.operator_model_id,
            metadata_duration=sample.metadata_duration,
            smla_duration=sample.operator_duration,
        )
        for sample in extract_operator_samples(profile_folder, PERF_CONFIG)
    ]


def collect_profile_samples(profile_root):
    """Extract measurements from all PROF directories without dropping runs."""
    samples = []
    for profile_folder in find_profile_folders(profile_root):
        folder_samples = extract_profile_samples(profile_folder)
        LOGGER.info(
            "%s: collected %d SparseFlashMla run(s)",
            profile_folder.name,
            len(folder_samples),
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


def _same_result_row(result_frame, existing_frame, index):
    return all(
        column in existing_frame.columns
        and _same_value(
            result_frame.at[index, column], existing_frame.at[index, column]
        )
        for column in result_frame.columns
    )


def _sample_key(sample):
    values = tuple(getattr(sample, column) for column in PERF_COLUMNS)
    return tuple(None if pd.isna(value) else value for value in values)


def _row_sample_key(row):
    return tuple(
        None if pd.isna(row.get(column)) else row.get(column) for column in PERF_COLUMNS
    )


def _repair_legacy_samples(existing, samples, profile_root):
    """Replay only the current capture when an old table contains surplus rows."""
    slot = ["profile_folder", "profile_run"]
    required = [*slot, "metadata_duration", "smla_duration"]
    if not set(required).issubset(existing.columns):
        return
    latest = existing.drop_duplicates(slot, keep="last")
    live = {
        (s.profile_folder, s.profile_run): s
        for s in samples
        if pd.notna(s.smla_duration)
    }
    active = latest[latest["profile_folder"].isin({folder for folder, _ in live})]
    unfiltered = replace(PERF_CONFIG, model_id_column="")

    if SMLA_MODEL_ID_COLUMN not in existing:
        # Older tables also saved step/compile invocations. Check their actual
        # durations in the surviving capture before clearing a reused folder.
        graph_folders = {
            s.profile_folder
            for s in live.values()
            if _is_graph_model_id(s.smla_model_id)
        }
        raw_values = {
            (s.profile_folder, s.profile_run, s.operator_duration)
            for folder in graph_folders
            for s in extract_operator_samples(Path(profile_root) / folder, unfiltered)
            if pd.notna(s.operator_duration)
        }
        raw_values.update(
            (s.profile_folder, s.profile_run, s.smla_duration)
            for s in live.values()
            if s.profile_folder in graph_folders
        )
        for i, row in active.iterrows():
            if (
                row["profile_folder"],
                row["profile_run"],
                row["smla_duration"],
            ) in raw_values:
                existing.loc[i, required] = pd.NA
        return

    orphan = active["metadata_duration"].notna() & active["smla_duration"].isna()
    # An extra metadata row can occupy a case in the middle, shifting later
    # samples. Replay the matching live invocations and retain archived rows.
    replay = orphan.any()
    existing.loc[active.index[orphan], PERF_COLUMNS] = pd.NA
    legacy_metadata = {}
    for i, row in active.iterrows():
        sample = live.get((row["profile_folder"], row["profile_run"]))
        if sample is None or not (
            _same_value(row["smla_duration"], sample.smla_duration)
            and _same_value(row[SMLA_MODEL_ID_COLUMN], sample.smla_model_id)
        ):
            continue
        if replay:
            existing.loc[i, PERF_COLUMNS] = pd.NA
        elif not _same_value(row["metadata_duration"], sample.metadata_duration):
            # Equal metadata and main counts could still pair the wrong rows.
            # Confirm the old positional value before changing an earlier run.
            folder = sample.profile_folder
            if folder not in legacy_metadata:
                legacy_metadata[folder] = {
                    s.profile_run: s.metadata_duration
                    for s in extract_operator_samples(
                        Path(profile_root) / folder, unfiltered
                    )
                }
            if _same_value(
                row["metadata_duration"], legacy_metadata[folder][sample.profile_run]
            ):
                existing.at[i, "metadata_duration"] = sample.metadata_duration


def merge_results_and_perf(result_frame, samples, existing_frame=None):
    """Map new profiler samples to pytest rows while preserving earlier runs."""
    merged = result_frame.copy()
    for column in PERF_COLUMNS:
        merged[column] = pd.NA

    if existing_frame is not None:
        shared_row_count = min(len(merged), len(existing_frame))
        for index in range(shared_row_count):
            if not _same_result_row(result_frame, existing_frame, index):
                continue
            for column in PERF_COLUMNS:
                if column in existing_frame.columns:
                    merged.at[index, column] = existing_frame.at[index, column]

    valid_indices = _valid_result_indices(merged)
    empty_indices = [
        index
        for index in valid_indices
        if pd.isna(merged.at[index, "metadata_duration"])
        and pd.isna(merged.at[index, "smla_duration"])
    ]
    existing_keys = {
        _row_sample_key(row)
        for _, row in merged.iterrows()
        if not (pd.isna(row["metadata_duration"]) and pd.isna(row["smla_duration"]))
    }
    new_samples = [
        sample for sample in samples if _sample_key(sample) not in existing_keys
    ]
    # msprof may reuse and overwrite the same PROF directory. In that case the
    # profile key can repeat, but newly appended pytest rows still need the
    # current measurements.
    if empty_indices and len(new_samples) < len(empty_indices):
        new_samples = samples[-len(empty_indices) :]

    for result_index, sample in zip(empty_indices, new_samples):
        for column in PERF_COLUMNS:
            merged.at[result_index, column] = getattr(sample, column)

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
            for column in PERF_COLUMNS:
                row[column] = getattr(sample, column)
            extra_rows.append(row)
        merged = pd.concat([merged, pd.DataFrame(extra_rows)], ignore_index=True)

    if existing_frame is not None and "smla_duration" in existing_frame:
        # A prior run may have more real measurements than accuracy rows, even
        # after its PROF directory has been archived or reused.
        historical = existing_frame.reindex(columns=merged.columns)
        extras = historical[
            historical["testcase_name"].isna() & historical["smla_duration"].notna()
        ]
        represented = {_row_sample_key(row) for _, row in merged.iterrows()}
        archived = [
            i for i, row in extras.iterrows() if _row_sample_key(row) not in represented
        ]
        if archived:
            merged = pd.concat([merged, extras.loc[archived]], ignore_index=True)

    return merged


def process_profiler_data(
    test_result_path,
    profile_root=".",
    output_path=None,
    delete_perf_files=False,
    testcase_paths=None,
    merge_into=None,
):
    """Merge all available profiler runs and save them atomically."""
    result_path = Path(test_result_path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Test result file does not exist: {result_path}")

    if testcase_paths is None:
        testcase_paths = (
            os.environ.get("QSAS_TESTCASE_PATH", "").strip(),
            os.environ.get("SMLA_PT_LOAD_PATH", "./data"),
        )
    if merge_into and output_path:
        raise ValueError("--merge_into and --output_path are mutually exclusive")
    if (
        merge_into
        or not output_path
        or Path(output_path).resolve() == result_path.resolve()
    ):
        return update_result_workbook(
            PERF_CONFIG,
            result_path,
            profile_root,
            merge_into or result_path,
            testcase_paths,
            PERF_COLUMNS,
            delete_perf_files=delete_perf_files,
        )
    result_frame = normalise_testcase_name_column(
        pd.read_excel(result_path), required=True
    )
    result_frame = replace_testcase_names_with_filenames(
        result_frame,
        testcase_paths,
        strip_pt_suffix=True,
    )
    profile_folders = find_profile_folders(profile_root)
    if not profile_folders:
        raise FileNotFoundError(f"No PROF directory found under: {profile_root}")

    samples = collect_profile_samples(profile_root)
    if not samples:
        raise ValueError("No SparseFlashMla performance data found in PROF directories")

    destination = Path(output_path)
    existing_frame = (
        normalise_testcase_name_column(pd.read_excel(destination), required=False)
        if destination.is_file()
        else None
    )
    if existing_frame is not None:
        existing_frame = replace_testcase_names_with_filenames(
            existing_frame,
            testcase_paths,
            strip_pt_suffix=True,
        )
        _repair_legacy_samples(existing_frame, samples, profile_root)
    merged = merge_results_and_perf(result_frame, samples, existing_frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.tmp.xlsx")
    merged.to_excel(temporary_path, index=False)
    os.replace(temporary_path, destination)

    LOGGER.info(
        "Saved %d profiler run(s) from %d folder(s) to %s",
        len(samples),
        len(profile_folders),
        destination,
    )

    if delete_perf_files:
        for profile_folder in profile_folders:
            shutil.rmtree(profile_folder)
            LOGGER.info("Deleted profiler directory: %s", profile_folder)

    return destination


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge all SparseFlashMla msprof data into one Excel table"
    )
    parser.add_argument(
        "--test_result_path",
        default="./result/smla_result.xlsx",
        help="pytest result Excel path",
    )
    parser.add_argument(
        "--profile_root",
        default=".",
        help="directory containing one or more PROF* folders",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output_path", help="write a separate performance export")
    output.add_argument(
        "--merge_into", help="fill matching cases in this accuracy workbook"
    )
    parser.add_argument(
        "--delete_perf_files",
        action="store_true",
        help="delete PROF* folders after a successful merge",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    try:
        process_profiler_data(
            test_result_path=args.test_result_path,
            profile_root=args.profile_root,
            output_path=args.output_path,
            merge_into=args.merge_into,
            delete_perf_files=args.delete_perf_files,
        )
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as error:
        LOGGER.error("Failed to process profiler data: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
