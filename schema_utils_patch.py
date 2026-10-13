"""
schema_utils_patch.py — 10 Federal Schema Drift Prevention Patch
================================================================
Patched version of functions from mongodb-partners/MongoDB_Fabric_Mirroring schema_utils.py.

PURPOSE:
  When MongoDB changes a field's type at the source (e.g., boolean -> string,
  double -> long), the mirroring app can produce parquet files whose column
  dtypes diverge from the pickled schema, causing SchemaMergeFailure in Fabric.

  This patch adds a final schema enforcement step in both `process_dataframe()`
  and `finalize_dataframe_for_parquet()` that FORCES every column to match the
  pickled schema dtype. Values that cannot be converted are set to null.

KNOWN REAL-WORLD CASES:
  - `latest`: boolean -> string (in _pages_versions, _cities_versions, etc.)
  - `primary_key_id`: double -> long (in unit-amenities)

HOW TO APPLY:
  See README_deploy.md in this directory.

UPSTREAM REF:
  https://github.com/mongodb-partners/MongoDB_Fabric_Mirroring (main branch)
  Patched against commit as of 2026-10-12.
"""

import logging

import numpy as np
import pandas as pd
try:
    import bson
    import bson.int64
except ImportError:
    bson = None

logger = logging.getLogger(__name__)

# All upstream app imports are LAZY — loaded inside functions at first use.
# This prevents import failures when the module is loaded at app startup
# before the mirror thread has initialized its dependencies.

_lazy_cache = {}


def _lazy_import(name):
    """Import a module lazily and cache it."""
    if name not in _lazy_cache:
        import importlib
        _lazy_cache[name] = importlib.import_module(name)
    return _lazy_cache[name]


def _get_schemas():
    return _lazy_import("schemas")


def _get_DTYPE_KEY():
    return _lazy_import("constants").DTYPE_KEY

# ---------------------------------------------------------------------------
# Re-import helpers from the original schema_utils that we call but do NOT
# modify. These are expected to exist in the module namespace at runtime.
# When deploying, either (a) monkey-patch the two functions into the existing
# module, or (b) replace the entire schema_utils.py with the merged file.
#
# The functions below reference these upstream symbols:
#   _is_null_value, _is_non_null_scalar, _needs_type_conversion,
#   _get_first_item, _apply_column_conversion, _cast_column_to_schema_dtype,
#   _typed_null_series, _resolve_string_schema_column,
#   init_column_schema, process_column_name,
#   TYPE_TO_CONVERT_FUNCTION_MAP, COLUMN_DTYPE_CONVERSION_MAP,
#   do_nothing, conversion_flag, current_column_name, table_name,
#   current_document_id, CustomJSONEncoder
#
# When using the monkey-patch approach (Option A in README_deploy.md),
# these are accessed via the original module.
# ---------------------------------------------------------------------------


# ===================================================================
# PATCH 1: enforce_schema_dtypes() — NEW FUNCTION
# ===================================================================

def enforce_schema_dtypes(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    [10FED PATCH] Final schema enforcement gate.

    After all upstream processing, iterate every column and force it to match
    the pickled schema dtype. If conversion fails, replace the entire column
    with typed nulls rather than risk a SchemaMergeFailure in Fabric.

    This function is ADDITIVE — it does not replace any upstream logic, it
    runs AFTER process_dataframe() and/or finalize_dataframe_for_parquet().
    """
    for col_name in df.columns:
        if col_name == "_id":
            continue

        schema_of_col = _get_schemas().get_table_column_schema(table_name, col_name)
        if not schema_of_col:
            continue

        expected_dtype = schema_of_col.get(_get_DTYPE_KEY())
        if expected_dtype is None:
            continue

        expected_dtype_str = str(expected_dtype)
        current_dtype_str = str(df[col_name].dtype)

        # Normalize for comparison: "Int64" vs "int64", "boolean" vs "bool", etc.
        if _dtypes_match(current_dtype_str, expected_dtype_str):
            continue

        logger.warning(
            "[10FED SCHEMA ENFORCEMENT] table=%s column=%s: "
            "current dtype '%s' != expected dtype '%s'. Forcing conversion.",
            table_name,
            col_name,
            current_dtype_str,
            expected_dtype_str,
        )

        try:
            df[col_name] = _force_convert_column(
                df[col_name], current_dtype_str, expected_dtype_str, col_name, table_name
            )
            logger.info(
                "[10FED SCHEMA ENFORCEMENT] table=%s column=%s: "
                "successfully converted to '%s'.",
                table_name,
                col_name,
                expected_dtype_str,
            )
        except Exception as exc:
            logger.critical(
                "[10FED SCHEMA ENFORCEMENT] DATA LOSS: table=%s column=%s: "
                "conversion to '%s' FAILED (%s). "
                "ENTIRE COLUMN (%d rows) replaced with typed nulls. "
                "This means all values for this column are lost in this batch.",
                table_name,
                col_name,
                expected_dtype_str,
                exc,
                len(df),
            )
            df[col_name] = _make_typed_null_series(len(df), expected_dtype_str)

    return df


def _dtypes_match(current: str, expected: str) -> bool:
    """Compare dtype strings accounting for common Pandas/PyArrow aliases."""
    if current == expected:
        return True
    # Normalize both to lowercase for comparison
    c = current.lower().strip()
    e = expected.lower().strip()
    if c == e:
        return True
    # "Int64" (nullable) and "int64" are different in Pandas but both are int64 semantically
    # However, for parquet schema purposes they produce different results.
    # We treat them as matching only if they are literally the same.
    # Common equivalences that are safe:
    equivalences = {
        ("float64", "float64"): True,
        ("string", "object"): True,
        ("object", "string"): True,
        ("string[pyarrow]", "object"): True,
        ("object", "string[pyarrow]"): True,
        ("string[pyarrow]", "string"): True,
        ("string", "string[pyarrow]"): True,
        ("datetime64[ms]", "datetime64[ns]"): True,
        ("datetime64[ns]", "datetime64[ms]"): True,
    }
    return equivalences.get((c, e), False)


def _force_convert_column(
    series: pd.Series,
    current_dtype: str,
    expected_dtype: str,
    col_name: str,
    table_name: str,
) -> pd.Series:
    """
    Force a series to the expected dtype, handling known problematic conversions.

    Known real-world drift cases:
      - boolean -> string (or vice versa)
      - double/float64 -> Int64/long (or vice versa)
      - object -> boolean/int/float
    """
    expected_lower = expected_dtype.lower()

    # --- CASE: expected is boolean ---
    if expected_lower == "boolean":
        return _force_to_boolean(series, col_name, table_name)

    # --- CASE: expected is Int64 (nullable integer) ---
    if expected_lower == "int64":
        # Check if it's the nullable "Int64" (capital I) which is what the schema uses
        return _force_to_int64(series, col_name, table_name)

    # --- CASE: expected is float64 ---
    if expected_lower in ("float64", "float64[pyarrow]"):
        return _force_to_float64(series, col_name, table_name)

    # --- CASE: expected is string/object ---
    if expected_lower in ("object", "string", "str", "string[pyarrow]"):
        return _force_to_string(series)

    # --- CASE: expected is datetime ---
    if expected_lower.startswith("datetime"):
        return _force_to_datetime(series)

    # --- FALLBACK: try direct astype ---
    return series.astype(expected_dtype)


def _force_to_boolean(series: pd.Series, col_name: str, table_name: str) -> pd.Series:
    """
    Convert any series to nullable boolean.

    Handles string->bool ("true"/"false"/"0"/"1"), numeric->bool (0/1),
    and sets everything else to pd.NA.
    """
    def _to_bool(val):
        if val is None or val is pd.NA:
            return pd.NA
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, np.integer, float, np.floating)):
            if val == 0:
                return False
            if val == 1:
                return True
            # Non-0/1 numeric -> null (ambiguous)
            return pd.NA
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no", ""):
                return False
            return pd.NA
        try:
            if pd.isna(val):
                return pd.NA
        except (TypeError, ValueError):
            pass
        # Unrecognized type -> null
        logger.debug(
            "[10FED] _force_to_boolean: cannot convert value of type %s for column %s. Setting null.",
            val,
            type(val).__name__,
            col_name,
        )
        return pd.NA

    converted = series.map(_to_bool)
    return converted.astype("boolean")


def _force_to_int64(series: pd.Series, col_name: str, table_name: str) -> pd.Series:
    """
    Convert any series to nullable Int64.

    Handles float->int (only if integer-valued), string->int, bool->int,
    bson types->int.

    Note: bson imported lazily inside function.
    Decimal128->int. Non-integer floats and unparseable strings become pd.NA.
    """
    def _to_int(val):
        if val is None or val is pd.NA:
            return pd.NA
        try:
            if pd.isna(val):
                return pd.NA
        except (TypeError, ValueError):
            pass
        if isinstance(val, (bool, np.bool_)):
            return int(val)
        if isinstance(val, bson.int64.Int64):
            return int(val)
        if isinstance(val, bson.Decimal128):
            dec = val.to_decimal()
            if dec == int(dec):
                return int(dec)
            return pd.NA
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            if np.isnan(val) or np.isinf(val):
                return pd.NA
            if float(val).is_integer():
                return int(val)
            # Non-integer float: cannot safely convert
            logger.debug(
                "[10FED] _force_to_int64: non-integer float value for column %s. Setting null.",
                val,
                col_name,
            )
            return pd.NA
        if isinstance(val, str):
            val = val.strip()
            if val == "":
                return pd.NA
            try:
                f = float(val)
                if f != f:  # NaN check
                    return pd.NA
                if float(f).is_integer():
                    return int(f)
                return pd.NA
            except (ValueError, OverflowError):
                return pd.NA
        return pd.NA

    converted = series.map(_to_int)
    return converted.astype("Int64")


def _force_to_float64(series: pd.Series, col_name: str, table_name: str) -> pd.Series:
    """
    Convert any series to float64.

    Handles int->float, string->float, bool->float, Decimal128->float.
    Unparseable values become NaN.
    """
    def _to_float(val):
        if val is None or val is pd.NA:
            return np.nan
        try:
            if pd.isna(val):
                return np.nan
        except (TypeError, ValueError):
            pass
        if isinstance(val, (bool, np.bool_)):
            return float(val)
        if isinstance(val, bson.Decimal128):
            return float(val.to_decimal())
        if isinstance(val, bson.int64.Int64):
            return float(int(val))
        if isinstance(val, (int, np.integer, float, np.floating)):
            return float(val)
        if isinstance(val, str):
            val = val.strip()
            if val == "":
                return np.nan
            try:
                return float(val)
            except (ValueError, OverflowError):
                return np.nan
        return np.nan

    converted = series.map(_to_float)
    return converted.astype("float64")


def _force_to_string(series: pd.Series) -> pd.Series:
    """Convert any series to string dtype."""
    def _to_str(val):
        if val is None or val is pd.NA:
            return pd.NA
        try:
            if pd.isna(val):
                return pd.NA
        except (TypeError, ValueError):
            pass
        if isinstance(val, (dict, list)):
            return json.dumps(val, default=str)
        return str(val)

    return series.map(_to_str).astype("string")


def _force_to_datetime(series: pd.Series) -> pd.Series:
    """Convert any series to datetime64[ms]."""
    series = pd.to_datetime(series, errors="coerce", utc=True)
    if hasattr(series.dt, "tz_localize"):
        try:
            series = series.dt.tz_localize(None)
        except TypeError:
            # Already tz-naive
            pass
    return series.astype("datetime64[ms]")


def _make_typed_null_series(length: int, dtype_str: str) -> pd.Series:
    """Create a series of the correct nullable dtype filled with nulls."""
    dtype_lower = dtype_str.lower()
    if dtype_lower == "boolean":
        return pd.Series([pd.NA] * length, dtype="boolean")
    if dtype_lower == "int64":
        return pd.Series([pd.NA] * length, dtype="Int64")
    if dtype_lower in ("float64", "float64[pyarrow]"):
        return pd.Series([np.nan] * length, dtype="float64")
    if dtype_lower.startswith("datetime"):
        return pd.Series([pd.NaT] * length, dtype="datetime64[ms]")
    return pd.Series([pd.NA] * length, dtype="string")


# ===================================================================
# PATCH 2: Monkey-patch installer
# ===================================================================

_PATCH_INSTALLED = False


def install_patch():
    """
    Monkey-patch the upstream schema_utils module to add schema enforcement.

    Call this ONCE at app startup (e.g., in function_app.py or __init__.py)
    BEFORE any sync operations begin.

    What it does:
    1. Wraps `process_dataframe()` to call `enforce_schema_dtypes()` after
       the original processing.
    2. Wraps `finalize_dataframe_for_parquet()` to call `enforce_schema_dtypes()`
       after the original finalization.
    """
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        logger.warning(
            "[10FED SCHEMA ENFORCEMENT] Patch already installed. Skipping."
        )
        return
    import schema_utils as upstream

    # Save references to the original functions
    _original_process_dataframe = upstream.process_dataframe
    _original_finalize_for_parquet = upstream.finalize_dataframe_for_parquet

    def patched_process_dataframe(table_name_param: str, df: pd.DataFrame):
        """Wrapper: run original process_dataframe, then enforce schema dtypes."""
        _original_process_dataframe(table_name_param, df)
        enforce_schema_dtypes(table_name_param, df)

    def patched_finalize_dataframe_for_parquet(table_name: str, df: pd.DataFrame) -> None:
        """Wrapper: run original finalize, then enforce schema dtypes."""
        _original_finalize_for_parquet(table_name, df)
        enforce_schema_dtypes(table_name, df)

    # Apply the patches
    upstream.process_dataframe = patched_process_dataframe
    upstream.finalize_dataframe_for_parquet = patched_finalize_dataframe_for_parquet

    _PATCH_INSTALLED = True
    logger.info(
        "[10FED SCHEMA ENFORCEMENT] Patch installed. "
        "process_dataframe and finalize_dataframe_for_parquet are now wrapped."
    )


# ===================================================================
# PATCH 3: Standalone enforcement (alternative to monkey-patch)
# ===================================================================

def enforce_parquet_before_write(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Standalone function to call right before writing a parquet file.

    Use this if you prefer explicit calls over monkey-patching.
    Can be inserted in init_sync.py and listening.py at the parquet write points.

    Example:
        from schema_utils_patch import enforce_parquet_before_write
        df = enforce_parquet_before_write(table_name, df)
        df.to_parquet(file_path)
    """
    return enforce_schema_dtypes(table_name, df)
