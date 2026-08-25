"""Deterministic audit of AQR's published monthly TSMOM factor returns.

This module does not construct signals or read any FTMOQuant market,
validation, or holdout data.  Its only return inputs are the two official AQR
workbooks named in :data:`UPDATED_SOURCE_URL` and :data:`ORIGINAL_SOURCE_URL`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.statistics import (
    StationaryBootstrapConfig,
    stationary_bootstrap_confidence_interval,
)

UPDATED_SOURCE_PAGE = (
    "https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Factors-Monthly"
)
UPDATED_SOURCE_URL = (
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
    "Time-Series-Momentum-Factors-Monthly.xlsx"
)
ORIGINAL_SOURCE_PAGE = (
    "https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data"
)
ORIGINAL_SOURCE_URL = (
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
    "Time-Series-Momentum-Original-Paper-Data.xlsx"
)
TERMS_URL = "https://www.aqr.com/Terms-of-Use"

UPDATED_FILENAME = "Time-Series-Momentum-Factors-Monthly.xlsx"
ORIGINAL_FILENAME = "Time-Series-Momentum-Original-Paper-Data.xlsx"
UPDATED_EXPECTED_SHA256 = (
    "33470930e2269c0d97be4732ec2d9c27ddbc69ac8133b059a263e27400263eeb"
)
ORIGINAL_EXPECTED_SHA256 = (
    "91bdbae6366ccb0693581b690236dc14862562a98ee83052c4f440f8b6ae0db8"
)

FACTOR_COLUMNS = (
    "all_assets",
    "equities",
    "currencies",
    "fixed_income",
    "commodities",
)
ASSET_CLASS_COLUMNS = FACTOR_COLUMNS[1:]
SOURCE_COLUMN_MAP = {
    "TSMOM": "all_assets",
    "TSMOM^EQ": "equities",
    "TSMOM^FX": "currencies",
    "TSMOM^FI": "fixed_income",
    "TSMOM^CM": "commodities",
}

PERIODS: Mapping[str, tuple[str, str | None]] = {
    "SOURCE_SAMPLE": ("1985-01", "2009-12"),
    "BRIDGE": ("2010-01", "2012-12"),
    "STRICT_POST_PUBLICATION": ("2013-01", None),
    "EARLY_POST_PUBLICATION": ("2013-01", "2019-12"),
    "RECENT": ("2020-01", None),
}
ANNUAL_COST_HURDLES = (0.0, 0.01, 0.02, 0.03)
PRIMARY_HURDLE = 0.02
SEVERE_HURDLE = 0.03
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_EXPECTED_BLOCK_LENGTH = 12
BOOTSTRAP_SEED = 20_260_825
BOOTSTRAP_ONE_SIDED_CONFIDENCE = 0.90
BOOTSTRAP_EQUIVALENT_TWO_SIDED_SIZE = 0.80

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REF = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")


class Xat1AqrAuditError(RuntimeError):
    """Raised when official-source identity or audit inputs are ambiguous."""


@dataclass(frozen=True, slots=True)
class GateInputs:
    strict_mean_2pct: float
    early_mean_2pct: float
    recent_mean_2pct: float
    strict_mean_3pct: float
    strict_asset_class_means_2pct: tuple[float, float, float, float]
    strict_bootstrap_lower_2pct: float


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: int
    description: str
    value: float | int
    threshold: str
    passed: bool


@dataclass(frozen=True, slots=True)
class WorkbookData:
    returns: pd.DataFrame
    raw_dates: tuple[str, ...]
    sheets: tuple[str, ...]
    factor_number_formats: tuple[str, ...]


def annual_drag_to_monthly(annual_drag: float) -> float:
    """Convert one of the frozen annual screening drags to a monthly drag."""

    if annual_drag not in ANNUAL_COST_HURDLES:
        raise Xat1AqrAuditError("annual drag is not one of the frozen hurdles")
    return annual_drag / 12.0


def detect_return_units(
    values: pd.DataFrame | np.ndarray[Any, np.dtype[np.float64]],
    *,
    number_formats: Sequence[str] | None = None,
) -> Literal["decimal", "percent"]:
    """Detect return units, failing closed when evidence is contradictory.

    Excel percentage formats are decisive: spreadsheet cells formatted with
    ``%`` store decimal returns.  The heuristic branch exists for synthetic
    and non-workbook validation only.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise Xat1AqrAuditError("return values must be finite and non-empty")
    if number_formats is not None:
        formats = tuple(number_formats)
        if not formats:
            raise Xat1AqrAuditError("factor number formats are missing")
        percent_formats = tuple("%" in item for item in formats)
        if all(percent_formats):
            return "decimal"
        if any(percent_formats):
            raise Xat1AqrAuditError("factor number formats give mixed unit evidence")
    maximum = float(np.max(np.abs(array)))
    if maximum <= 0.50:
        return "decimal"
    if 2.0 <= maximum <= 100.0:
        return "percent"
    raise Xat1AqrAuditError("return units are ambiguous")


def validate_factor_frame(
    frame: pd.DataFrame,
    *,
    units: Literal["decimal", "percent"],
) -> pd.DataFrame:
    """Validate exact schema and a complete, unique monthly calendar."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise Xat1AqrAuditError("factor frame must be a non-empty DataFrame")
    if tuple(frame.columns) != FACTOR_COLUMNS:
        raise Xat1AqrAuditError("wrong factor columns or missing primary factor")
    if not frame.columns.is_unique:
        raise Xat1AqrAuditError("duplicate factor columns")
    try:
        index = _to_period_index(frame.index)
        array = frame.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise Xat1AqrAuditError("malformed date or non-numeric factor value") from error
    if not np.isfinite(array).all():
        raise Xat1AqrAuditError("missing or non-finite factor value")
    if not index.is_unique:
        raise Xat1AqrAuditError("duplicate month")
    if not index.is_monotonic_increasing:
        raise Xat1AqrAuditError("months must be strictly increasing")
    expected = pd.period_range(index[0], index[-1], freq="M")
    if not index.equals(expected):
        raise Xat1AqrAuditError("missing month")
    if units == "percent":
        array /= 100.0
    elif units != "decimal":
        raise Xat1AqrAuditError("units must be decimal or percent")
    if float(np.max(np.abs(array))) > 1.0:
        raise Xat1AqrAuditError("decimal return magnitude exceeds 100%")
    return pd.DataFrame(array, index=index, columns=FACTOR_COLUMNS)


def load_aqr_workbook(path: Path) -> WorkbookData:
    """Read the exact AQR factor sheet using only standard OOXML structures."""

    try:
        with ZipFile(path) as archive:
            sheets = _sheet_targets(archive)
            factor_sheet_names = [
                name for name in sheets if name.casefold() == "tsmom factors"
            ]
            if len(factor_sheet_names) != 1:
                raise Xat1AqrAuditError("TSMOM Factors sheet is missing")
            sheet_names = tuple(sheets)
            if "disclosures" not in {name.casefold() for name in sheet_names}:
                raise Xat1AqrAuditError("required description/disclosure tab missing")
            strings = _shared_strings(archive)
            formats = _style_number_formats(archive)
            rows = _worksheet_rows(archive, sheets[factor_sheet_names[0]], strings)
    except (BadZipFile, KeyError, ElementTree.ParseError, OSError) as error:
        raise Xat1AqrAuditError(f"invalid OOXML workbook: {path}") from error

    header_position = _find_header(rows)
    header = rows[header_position]
    source_by_column = {
        column: cast(str, cell["value"])
        for column, cell in header.items()
        if cell["value"] in SOURCE_COLUMN_MAP
    }
    if set(source_by_column.values()) != set(SOURCE_COLUMN_MAP):
        raise Xat1AqrAuditError("wrong factor columns")
    factor_formats: set[str] = set()
    records: list[dict[str, Any]] = []
    raw_dates: list[str] = []
    for row in rows[header_position + 1 :]:
        populated_factors = [
            row.get(column, {}).get("value") for column in source_by_column
        ]
        blank = [value is None or value == "" for value in populated_factors]
        if all(blank):
            continue
        if any(blank):
            raise Xat1AqrAuditError("missing factor value")
        date_value = row.get("A", {}).get("value")
        if not isinstance(date_value, (float, int)):
            raise Xat1AqrAuditError("malformed workbook date")
        raw_date = _excel_date(float(date_value))
        record: dict[str, Any] = {"month": raw_date.strftime("%Y-%m")}
        raw_dates.append(raw_date.date().isoformat())
        for column, source_name in source_by_column.items():
            value = row[column]["value"]
            if not isinstance(value, (float, int)):
                raise Xat1AqrAuditError("non-numeric factor value")
            style = int(row[column]["style"])
            try:
                factor_formats.add(formats[style])
            except IndexError as error:
                raise Xat1AqrAuditError("invalid factor cell style") from error
            record[SOURCE_COLUMN_MAP[source_name]] = float(value)
        records.append(record)
    if not records:
        raise Xat1AqrAuditError("factor sheet contains no observations")
    raw = pd.DataFrame.from_records(records).set_index("month")
    raw = raw.loc[:, FACTOR_COLUMNS]
    units = detect_return_units(raw, number_formats=tuple(sorted(factor_formats)))
    returns = validate_factor_frame(raw, units=units)
    return WorkbookData(
        returns=returns,
        raw_dates=tuple(raw_dates),
        sheets=sheet_names,
        factor_number_formats=tuple(sorted(factor_formats)),
    )


def slice_frozen_period(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    """Slice one and only one preregistered calendar period."""

    if period not in PERIODS:
        raise Xat1AqrAuditError("period is not frozen")
    start, frozen_end = PERIODS[period]
    end = frozen_end or str(frame.index[-1])
    result = frame.loc[pd.Period(start, "M") : pd.Period(end, "M")]
    if result.empty:
        raise Xat1AqrAuditError(f"frozen period has no observations: {period}")
    expected_start = pd.Period(start, "M")
    if result.index[0] != expected_start:
        raise Xat1AqrAuditError(f"frozen period start is unavailable: {period}")
    if frozen_end is not None and result.index[-1] != pd.Period(frozen_end, "M"):
        raise Xat1AqrAuditError(f"frozen period end is unavailable: {period}")
    return result


def metric_summary(
    returns: pd.Series,
    *,
    bootstrap_gross_monthly: tuple[float, float],
    annual_drag: float,
) -> dict[str, Any]:
    """Compute the frozen monthly-return diagnostics."""

    monthly_drag = annual_drag_to_monthly(annual_drag)
    values = returns.to_numpy(dtype=np.float64) - monthly_drag
    if np.any(values <= -1.0):
        raise Xat1AqrAuditError("cost hurdle creates a return at or below -100%")
    wealth = np.cumprod(1.0 + values)
    wealth_with_origin = np.concatenate(([1.0], wealth))
    drawdowns = wealth_with_origin / np.maximum.accumulate(wealth_with_origin) - 1.0
    volatility = float(np.std(values, ddof=1) * math.sqrt(12.0))
    mean = float(np.mean(values) * 12.0)
    worst_position = int(np.argmin(values))
    best_position = int(np.argmax(values))
    lower_gross, upper_gross = bootstrap_gross_monthly
    return {
        "months": int(len(values)),
        "arithmetic_annualized_mean": mean,
        "geometric_annual_return": float(wealth[-1] ** (12.0 / len(values)) - 1.0),
        "annualized_volatility": volatility,
        "sharpe_zero_cash": None if volatility == 0.0 else mean / volatility,
        "positive_month_fraction": float(np.mean(values > 0.0)),
        "maximum_drawdown": float(np.min(drawdowns)),
        "skewness": float(pd.Series(values).skew()),
        "worst_month": str(returns.index[worst_position]),
        "worst_month_return": float(values[worst_position]),
        "best_month": str(returns.index[best_position]),
        "best_month_return": float(values[best_position]),
        "cumulative_growth_of_1": float(wealth[-1]),
        "stationary_bootstrap_mean": {
            "lower_one_sided_90_annualized": float((lower_gross - monthly_drag) * 12.0),
            "upper_one_sided_90_annualized": float((upper_gross - monthly_drag) * 12.0),
            "equivalent_central_interval_size": (BOOTSTRAP_EQUIVALENT_TWO_SIDED_SIZE),
        },
    }


def evaluate_gates(inputs: GateInputs) -> tuple[GateResult, ...]:
    """Evaluate only the six frozen gates using the supplied all-asset inputs."""

    positive_classes = sum(
        value > 0.0 for value in inputs.strict_asset_class_means_2pct
    )
    return (
        GateResult(
            1,
            "strict all-assets mean after 2% hurdle",
            inputs.strict_mean_2pct,
            "> 0",
            inputs.strict_mean_2pct > 0.0,
        ),
        GateResult(
            2,
            "early all-assets mean after 2% hurdle",
            inputs.early_mean_2pct,
            "> 0",
            inputs.early_mean_2pct > 0.0,
        ),
        GateResult(
            3,
            "recent all-assets mean after 2% hurdle",
            inputs.recent_mean_2pct,
            "> 0",
            inputs.recent_mean_2pct > 0.0,
        ),
        GateResult(
            4,
            "strict all-assets mean after 3% hurdle",
            inputs.strict_mean_3pct,
            "> 0",
            inputs.strict_mean_3pct > 0.0,
        ),
        GateResult(
            5,
            "positive strict asset classes after 2% hurdle",
            positive_classes,
            ">= 3",
            positive_classes >= 3,
        ),
        GateResult(
            6,
            "strict all-assets bootstrap lower bound after 2% hurdle",
            inputs.strict_bootstrap_lower_2pct,
            "> 0",
            inputs.strict_bootstrap_lower_2pct > 0.0,
        ),
    )


def reconcile_original(updated: pd.DataFrame, original: pd.DataFrame) -> dict[str, Any]:
    """Reconcile exact frozen overlap without rounding or date adjustment."""

    overlap_updated = updated.loc[pd.Period("1985-01", "M") : pd.Period("2009-12", "M")]
    overlap_original = original.loc[
        pd.Period("1985-01", "M") : pd.Period("2009-12", "M")
    ]
    if not overlap_updated.index.equals(overlap_original.index):
        raise Xat1AqrAuditError("updated/original overlap months are not identical")
    result: dict[str, Any] = {
        "period": "1985-01 through 2009-12",
        "months": int(len(overlap_updated)),
        "factors": {},
        "documented_revision_context": (
            "AQR says the updated portfolio is based on the original paper but "
            "may differ in data sources and methodology; AQR reconstructs the "
            "full history each time returns are updated."
        ),
    }
    for factor in FACTOR_COLUMNS:
        left = overlap_updated[factor].to_numpy(dtype=np.float64)
        right = overlap_original[factor].to_numpy(dtype=np.float64)
        difference = left - right
        differing = difference != 0.0
        result["factors"][factor] = {
            "exact_equality": bool(np.array_equal(left, right)),
            "exact_equal_months": int(np.sum(~differing)),
            "months_with_differences": int(np.sum(differing)),
            "difference_months": [
                str(month)
                for month, is_different in zip(
                    overlap_updated.index, differing, strict=True
                )
                if is_different
            ],
            "maximum_absolute_difference": float(np.max(np.abs(difference))),
            "correlation": float(np.corrcoef(left, right)[0, 1]),
        }
    return result


def run_audit(
    updated_path: Path,
    original_path: Path,
    *,
    acquired_at_utc: Mapping[str, str],
) -> dict[str, Any]:
    """Run the full fixed audit and return a JSON-compatible document."""

    updated_sha = _sha256_file(updated_path)
    original_sha = _sha256_file(original_path)
    if updated_sha != UPDATED_EXPECTED_SHA256:
        raise Xat1AqrAuditError("updated workbook SHA differs from frozen acquisition")
    if original_sha != ORIGINAL_EXPECTED_SHA256:
        raise Xat1AqrAuditError("original workbook SHA differs from frozen acquisition")
    updated_workbook = load_aqr_workbook(updated_path)
    original_workbook = load_aqr_workbook(original_path)
    updated = updated_workbook.returns
    original = original_workbook.returns
    if updated.index[0] != pd.Period("1985-01", "M"):
        raise Xat1AqrAuditError("updated data do not begin 1985-01")
    if original.index[0] != pd.Period("1985-01", "M") or original.index[
        -1
    ] != pd.Period("2009-12", "M"):
        raise Xat1AqrAuditError("original-paper data have wrong boundaries")

    periods: dict[str, Any] = {}
    gross_bootstrap: dict[tuple[str, str], tuple[float, float]] = {}
    for period_name in PERIODS:
        period_frame = slice_frozen_period(updated, period_name)
        period_report: dict[str, Any] = {
            "start": str(period_frame.index[0]),
            "end": str(period_frame.index[-1]),
            "factors": {},
        }
        for factor in FACTOR_COLUMNS:
            series = period_frame[factor].rename(f"{period_name}:{factor}:gross")
            bootstrap = stationary_bootstrap_confidence_interval(
                series,
                StationaryBootstrapConfig(
                    block_size=BOOTSTRAP_EXPECTED_BLOCK_LENGTH,
                    repetitions=BOOTSTRAP_REPETITIONS,
                    seed=BOOTSTRAP_SEED,
                    confidence_level=BOOTSTRAP_EQUIVALENT_TWO_SIDED_SIZE,
                    method="percentile",
                ),
            )
            gross_bootstrap[(period_name, factor)] = (
                bootstrap.lower_bound,
                bootstrap.upper_bound,
            )
            factor_report: dict[str, Any] = {}
            for hurdle in ANNUAL_COST_HURDLES:
                factor_report[_hurdle_label(hurdle)] = metric_summary(
                    period_frame[factor],
                    bootstrap_gross_monthly=gross_bootstrap[(period_name, factor)],
                    annual_drag=hurdle,
                )
            period_report["factors"][factor] = factor_report
        period_report["asset_class_correlation_gross"] = _jsonable_frame(
            period_frame.loc[:, ASSET_CLASS_COLUMNS].corr()
        )
        period_report["breadth"] = _breadth(period_report["factors"])
        periods[period_name] = period_report

    strict = periods["STRICT_POST_PUBLICATION"]["factors"]
    early = periods["EARLY_POST_PUBLICATION"]["factors"]
    recent = periods["RECENT"]["factors"]
    gates = evaluate_gates(
        GateInputs(
            strict_mean_2pct=strict["all_assets"]["2pct"]["arithmetic_annualized_mean"],
            early_mean_2pct=early["all_assets"]["2pct"]["arithmetic_annualized_mean"],
            recent_mean_2pct=recent["all_assets"]["2pct"]["arithmetic_annualized_mean"],
            strict_mean_3pct=strict["all_assets"]["3pct"]["arithmetic_annualized_mean"],
            strict_asset_class_means_2pct=cast(
                tuple[float, float, float, float],
                tuple(
                    strict[factor]["2pct"]["arithmetic_annualized_mean"]
                    for factor in ASSET_CLASS_COLUMNS
                ),
            ),
            strict_bootstrap_lower_2pct=strict["all_assets"]["2pct"][
                "stationary_bootstrap_mean"
            ]["lower_one_sided_90_annualized"],
        )
    )
    decision = (
        "PROCEED_FREE_IMPLEMENTATION_FEASIBILITY"
        if all(item.passed for item in gates)
        else "RETIRE_XAT1"
    )
    module_path = Path(__file__).resolve()
    return {
        "schema": "ftmoquant.xat1-aqr-post-publication-factor-audit",
        "version": 1,
        "decision": decision,
        "governance": {
            "factor_returns_only": True,
            "ftmo_strategy_backtest": False,
            "existing_validation_or_holdout_accessed": False,
            "holdout_consequence": (
                "2020-2025 are no longer pristine evidence for broad cross-asset "
                "TSMOM, though they may remain unseen for a future specific "
                "FTMOQuant implementation."
            ),
        },
        "sources": {
            "updated": _source_record(
                updated_path,
                updated_sha,
                UPDATED_SOURCE_PAGE,
                UPDATED_SOURCE_URL,
                acquired_at_utc["updated"],
            ),
            "original": _source_record(
                original_path,
                original_sha,
                ORIGINAL_SOURCE_PAGE,
                ORIGINAL_SOURCE_URL,
                acquired_at_utc["original"],
            ),
            "terms_url": TERMS_URL,
        },
        "dataset_identity": {
            "page_version_date": "2026-05-29",
            "latest_included_month": str(updated.index[-1]),
            "observation_count": int(len(updated)),
            "raw_date_first": updated_workbook.raw_dates[0],
            "raw_date_last": updated_workbook.raw_dates[-1],
            "date_convention": (
                "source business-month-end dates normalized to calendar month"
            ),
            "factor_columns": list(FACTOR_COLUMNS),
            "source_factor_labels": SOURCE_COLUMN_MAP,
            "all_assets_aggregate_supplied": True,
            "units": "decimal values displayed with Excel percentage formats",
            "factor_number_formats": updated_workbook.factor_number_formats,
            "missing_values": 0,
            "duplicate_months": 0,
            "missing_months": 0,
            "workbook_tabs": updated_workbook.sheets,
            "return_description": "monthly excess returns of long/short factors",
            "cost_status": (
                "not explicitly identified by AQR as gross or net of transaction "
                "costs; no separate net-of-cost series is supplied"
            ),
            "construction": (
                "12-month TSMOM, 1-month holding period, 58 liquid instruments; "
                "updated construction is paper-based but AQR warns data sources "
                "and methodology can differ and full history is reconstructed"
            ),
            "terms_and_restrictions": (
                "informational only, no offer/advice, no accuracy warranty, past "
                "performance warning; AQR website Terms of Use restrict copying, "
                "modification, distribution, transfer, licensing, or publication "
                "without prior written consent, so raw workbooks remain outside Git"
            ),
        },
        "frozen_specification": {
            "periods": PERIODS,
            "annual_cost_hurdles": ANNUAL_COST_HURDLES,
            "primary_hurdle": PRIMARY_HURDLE,
            "severe_hurdle": SEVERE_HURDLE,
            "bootstrap": {
                "procedure": "arch StationaryBootstrap percentile mean interval",
                "resamples": BOOTSTRAP_REPETITIONS,
                "expected_block_length_months": BOOTSTRAP_EXPECTED_BLOCK_LENGTH,
                "seed": BOOTSTRAP_SEED,
                "one_sided_confidence": BOOTSTRAP_ONE_SIDED_CONFIDENCE,
                "equivalent_two_sided_interval_size": (
                    BOOTSTRAP_EQUIVALENT_TWO_SIDED_SIZE
                ),
            },
        },
        "original_data_reconciliation": reconcile_original(updated, original),
        "period_results": periods,
        "calendar_year_returns": _calendar_year_returns(updated),
        "rolling_36_month_diagnostics": _rolling_diagnostics(updated),
        "publication_decay": {
            "SOURCE_SAMPLE_minus_STRICT_POST_PUBLICATION": _period_comparison(
                periods, "SOURCE_SAMPLE", "STRICT_POST_PUBLICATION"
            ),
            "EARLY_POST_PUBLICATION_minus_RECENT": _period_comparison(
                periods, "EARLY_POST_PUBLICATION", "RECENT"
            ),
        },
        "gates": [asdict(item) for item in gates],
        "implementation": {
            "git_head": _git_head(module_path.parents[3]),
            "module_path": str(module_path),
            "module_sha256": _sha256_file(module_path),
        },
    }


def write_once_artifact(output_dir: Path, document: Mapping[str, Any]) -> Path:
    """Create a deterministic result directory and refuse all overwrites."""

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "audit.json"
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    return output


def _bootstrap_gross(
    frame: pd.DataFrame, period: str, factor: str
) -> tuple[float, float]:
    series = slice_frozen_period(frame, period)[factor].rename(
        f"{period}:{factor}:gross"
    )
    result = stationary_bootstrap_confidence_interval(
        series,
        StationaryBootstrapConfig(
            block_size=BOOTSTRAP_EXPECTED_BLOCK_LENGTH,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
            confidence_level=BOOTSTRAP_EQUIVALENT_TWO_SIDED_SIZE,
            method="percentile",
        ),
    )
    return result.lower_bound, result.upper_bound


def _breadth(factor_reports: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for hurdle in ANNUAL_COST_HURDLES:
        label = _hurdle_label(hurdle)
        means = {
            factor: factor_reports[factor][label]["arithmetic_annualized_mean"]
            for factor in ASSET_CLASS_COLUMNS
        }
        result[label] = {
            "asset_class_annualized_means": means,
            "positive_asset_classes": sum(value > 0.0 for value in means.values()),
        }
    return result


def _calendar_year_returns(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    years = frame.index.year
    for factor in FACTOR_COLUMNS:
        factor_result: dict[str, Any] = {}
        for hurdle in ANNUAL_COST_HURDLES:
            adjusted = frame[factor] - annual_drag_to_monthly(hurdle)
            compounded = (1.0 + adjusted).groupby(years).prod() - 1.0
            factor_result[_hurdle_label(hurdle)] = {
                str(year): float(value) for year, value in compounded.items()
            }
        result[factor] = factor_result
    return result


def _rolling_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for factor in FACTOR_COLUMNS:
        factor_result: dict[str, Any] = {}
        for hurdle in ANNUAL_COST_HURDLES:
            adjusted = frame[factor] - annual_drag_to_monthly(hurdle)
            rolling_mean = adjusted.rolling(36).mean() * 12.0
            rolling_vol = adjusted.rolling(36).std(ddof=1) * math.sqrt(12.0)
            rolling_sharpe = rolling_mean / rolling_vol
            factor_result[_hurdle_label(hurdle)] = {
                str(month): {
                    "arithmetic_annualized_mean": float(rolling_mean.loc[month]),
                    "sharpe_zero_cash": float(rolling_sharpe.loc[month]),
                }
                for month in frame.index[35:]
            }
        result[factor] = factor_result
    return result


def _period_comparison(
    periods: Mapping[str, Any], left_name: str, right_name: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = (
        "arithmetic_annualized_mean",
        "annualized_volatility",
        "sharpe_zero_cash",
        "maximum_drawdown",
        "positive_month_fraction",
    )
    for factor in FACTOR_COLUMNS:
        left = periods[left_name]["factors"][factor]["gross"]
        right = periods[right_name]["factors"][factor]["gross"]
        result[factor] = {
            "left_period": left_name,
            "right_period": right_name,
            "left_minus_right": {
                field: float(left[field] - right[field]) for field in fields
            },
            "left": {field: left[field] for field in fields},
            "right": {field: right[field] for field in fields},
        }
    return result


def _source_record(
    path: Path, sha256: str, page: str, url: str, acquired_at: str
) -> dict[str, Any]:
    return {
        "official_page": page,
        "official_download_url": url,
        "filename": path.name,
        "external_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256,
        "downloaded_at_utc": acquired_at,
        "committed_to_git": False,
    }


def _hurdle_label(hurdle: float) -> str:
    return "gross" if hurdle == 0.0 else f"{int(hurdle * 100)}pct"


def _to_period_index(index: pd.Index) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        if index.freqstr != "M":
            raise ValueError("dates must have monthly frequency")
        return index.copy()
    if any(not isinstance(value, str) for value in index):
        raise ValueError("dates must be YYYY-MM strings or a monthly PeriodIndex")
    values = cast(Sequence[str], index.tolist())
    if any(
        re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", value) is None for value in values
    ):
        raise ValueError("malformed date")
    return pd.PeriodIndex(values, freq="M")


def _excel_date(serial: float) -> datetime:
    if not math.isfinite(serial) or serial < 1.0 or serial != math.floor(serial):
        raise Xat1AqrAuditError("malformed Excel date serial")
    return datetime(1899, 12, 30, tzinfo=UTC) + timedelta(days=serial)


def _column(reference: str) -> str:
    match = _CELL_REF.fullmatch(reference)
    if match is None:
        raise Xat1AqrAuditError(f"malformed cell reference: {reference}")
    return match.group(1)


def _shared_strings(archive: ZipFile) -> tuple[str, ...]:
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return tuple(
        "".join(text.text or "" for text in item.iter(f"{{{_MAIN_NS}}}t"))
        for item in root.findall(f"{{{_MAIN_NS}}}si")
    )


def _style_number_formats(archive: ZipFile) -> tuple[str, ...]:
    root = ElementTree.fromstring(archive.read("xl/styles.xml"))
    custom = {
        int(item.attrib["numFmtId"]): item.attrib["formatCode"]
        for item in root.findall(f"{{{_MAIN_NS}}}numFmts/{{{_MAIN_NS}}}numFmt")
    }
    built_in = {9: "0%", 10: "0.00%", 11: "0.00E+00", 14: "mm-dd-yy"}
    cell_xfs = root.find(f"{{{_MAIN_NS}}}cellXfs")
    if cell_xfs is None:
        raise Xat1AqrAuditError("cell styles are missing")
    return tuple(
        custom.get(
            int(item.attrib.get("numFmtId", "0")),
            built_in.get(int(item.attrib.get("numFmtId", "0")), "General"),
        )
        for item in cell_xfs.findall(f"{{{_MAIN_NS}}}xf")
    )


def _sheet_targets(archive: ZipFile) -> dict[str, str]:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib["Id"]: item.attrib["Target"].lstrip("/")
        for item in relationships.findall(f"{{{_PKG_REL_NS}}}Relationship")
    }
    result: dict[str, str] = {}
    for sheet in workbook.findall(f"{{{_MAIN_NS}}}sheets/{{{_MAIN_NS}}}sheet"):
        target = targets[sheet.attrib[f"{{{_REL_NS}}}id"]]
        result[sheet.attrib["name"]] = (
            target if target.startswith("xl/") else f"xl/{target}"
        )
    return result


def _worksheet_rows(
    archive: ZipFile, target: str, strings: Sequence[str]
) -> list[dict[str, dict[str, Any]]]:
    root = ElementTree.fromstring(archive.read(target))
    result: list[dict[str, dict[str, Any]]] = []
    for row in root.findall(f"{{{_MAIN_NS}}}sheetData/{{{_MAIN_NS}}}row"):
        values: dict[str, dict[str, Any]] = {}
        for cell in row.findall(f"{{{_MAIN_NS}}}c"):
            reference = cell.attrib.get("r", "")
            column = _column(reference)
            value_node = cell.find(f"{{{_MAIN_NS}}}v")
            value: str | float | None = None
            if value_node is not None and value_node.text is not None:
                if cell.attrib.get("t") == "s":
                    value = strings[int(value_node.text)]
                else:
                    value = float(value_node.text)
            values[column] = {
                "value": value,
                "style": cell.attrib.get("s", "0"),
                "reference": reference,
            }
        result.append(values)
    return result


def _find_header(rows: Sequence[Mapping[str, Mapping[str, Any]]]) -> int:
    expected = set(SOURCE_COLUMN_MAP)
    matches = [
        position
        for position, row in enumerate(rows)
        if {cell.get("value") for cell in row.values()} & expected == expected
    ]
    if len(matches) != 1:
        raise Xat1AqrAuditError("factor header row is missing or ambiguous")
    return matches[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _jsonable_frame(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(column): {
            str(index): float(frame.loc[index, column]) for index in frame.index
        }
        for column in frame.columns
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = cast(Path, args.data_root)
    document = run_audit(
        root / UPDATED_FILENAME,
        root / ORIGINAL_FILENAME,
        acquired_at_utc={
            "updated": "2026-08-25T16:06:47Z",
            "original": "2026-08-25T16:06:53Z",
        },
    )
    output = write_once_artifact(cast(Path, args.output_dir), document)
    print(f"{document['decision']}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
