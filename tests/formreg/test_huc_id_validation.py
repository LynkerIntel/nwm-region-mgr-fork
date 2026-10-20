"""Unit tests for select_formulation's nested-HUC-ID handling.

Guards the assumption behind the 'upscaling' basin_fill_method (and any 'huc_level' coarser
than a hydrofabric's native/finest spatial unit): that the configured 'huc12' id column holds
nested, 12-digit numeric HUC codes where a coarser unit's ID is the digit-prefix of its
children's IDs. Without this guard, a non-conforming id column would be silently truncated
into spatial groupings with no real geographic meaning.
"""

import pandas as pd
import pytest

from nwm_region_mgr.formreg.select_formulation import (
    _compute_huc_id_column,
    _normalize_numeric_huc_artifacts,
    _validate_nested_huc_ids,
)


@pytest.mark.unit
def test_conforming_nested_huc_ids_pass():
    """12-digit numeric codes (optionally with nulls) pass without error."""
    values = pd.Series(["010100020101", "010100020102", None])
    _validate_nested_huc_ids(values, "huc_12")  # should not raise


@pytest.mark.unit
def test_non_numeric_ids_raise_with_actionable_message():
    """Non-numeric or wrong-length IDs raise, naming the column and the fix."""
    values = pd.Series(["cat-1", "cat-2", "010100020101"])
    with pytest.raises(ValueError, match="nested 12-digit numeric HUC codes") as exc_info:
        _validate_nested_huc_ids(values, "huc_12")
    assert "huc_12" in str(exc_info.value)
    assert "nearest-neighbor" in str(exc_info.value)


@pytest.mark.unit
def test_wrong_length_numeric_ids_raise():
    """A purely numeric but non-12-digit code is still rejected (not truly nested HUC12)."""
    values = pd.Series(["0101000201", "010100020101"])  # 10 digits, then 12 digits
    with pytest.raises(ValueError, match="nested 12-digit numeric HUC codes"):
        _validate_nested_huc_ids(values, "huc_12")


@pytest.mark.unit
def test_all_null_column_is_skipped():
    """An all-null column has nothing to validate and should not raise."""
    values = pd.Series([None, None], dtype=object)
    _validate_nested_huc_ids(values, "huc_12")  # should not raise


# -------------------------------_normalize_numeric_huc_artifacts-------------------------------
@pytest.mark.unit
def test_normalize_strips_trailing_dot_zero_from_float_dtype_artifact():
    """A huc12 column stored upstream as float64 gains a trailing '.0' when read back as
    text; that must be stripped so a genuinely nested HUC12 value isn't misdiagnosed.
    """
    values = pd.Series(["010100020101.0", "010200020101.0"])
    fixed = _normalize_numeric_huc_artifacts(values)
    assert fixed.tolist() == ["010100020101", "010200020101"]
    _validate_nested_huc_ids(fixed, "huc_12")  # should not raise


@pytest.mark.unit
def test_normalize_zero_pads_int_dtype_artifact_missing_leading_zero():
    """A huc12 column stored upstream as int64 loses a leading zero (e.g. VPU 01's prefix);
    zero-padding back to 12 digits recovers a genuinely nested HUC12 value.
    """
    values = pd.Series(["10100020101"])  # 11 digits: lost the leading '0'
    fixed = _normalize_numeric_huc_artifacts(values)
    assert fixed.tolist() == ["010100020101"]
    _validate_nested_huc_ids(fixed, "huc_12")  # should not raise


@pytest.mark.unit
def test_normalize_leaves_genuinely_non_nested_ids_untouched():
    """A custom hydrofabric's non-numeric native IDs are not numeric-dtype artifacts and
    must be left alone, so they're still correctly rejected by validation as non-conforming.
    """
    values = pd.Series(["cat-1", "cat-2"])
    fixed = _normalize_numeric_huc_artifacts(values)
    assert fixed.tolist() == ["cat-1", "cat-2"]
    with pytest.raises(ValueError, match="nested 12-digit numeric HUC codes"):
        _validate_nested_huc_ids(fixed, "huc_12")


@pytest.mark.unit
def test_normalize_preserves_null_entries():
    """Null entries must be preserved as null, not stringified to 'nan'/'None'."""
    values = pd.Series(["010100020101.0", None])
    fixed = _normalize_numeric_huc_artifacts(values)
    assert fixed.iloc[0] == "010100020101"
    assert pd.isna(fixed.iloc[1])


# -------------------------------_compute_huc_id_column-------------------------------
@pytest.mark.unit
def test_compute_huc_id_huc_digit_12_is_a_true_no_op():
    """huc_digit=12 (huc_level='huc12') must not truncate a native ID to 12 characters --
    doing so would silently collide two divides that share a 12-character prefix.
    """
    values = pd.Series(["catchment-0001-upper", "catchment-0001-lower", "catchment-0002"])
    result = _compute_huc_id_column(values, huc_digit=12)
    assert result.tolist() == values.tolist()
    assert len(set(result)) == 3  # no collision between the two 12-char-prefix-sharing IDs


@pytest.mark.unit
def test_compute_huc_id_unaffected_for_true_huc12_codes():
    """A true 12-digit HUC12 code is unaffected either way (no behavior change for NWM)."""
    values = pd.Series(["010100020101", "010100020102"])
    assert _compute_huc_id_column(values, huc_digit=12).tolist() == values.tolist()


@pytest.mark.unit
def test_compute_huc_id_coarser_level_still_truncates():
    """A coarser level (e.g. huc8) still truncates as before -- only huc_digit>=12 is a no-op."""
    values = pd.Series(["010100020101", "010100020102"])
    result = _compute_huc_id_column(values, huc_digit=8)
    assert result.tolist() == ["01010002", "01010002"]
