"""Regression tests proving parreg works against a renamed/custom hydrofabric schema.

These exercise the same code paths as tests/parreg/test_parreg_process_config.py, but with
every configurable id column renamed away from its default ('div_id'/'gage_id'/'vpu_id'), and
with an attribute dataset under a fully custom name. If any code on these paths still had a
hardcoded literal bypassing the configured id_col (the original motivation for these changes),
it would be exposed here as a KeyError/empty result rather than only in a config using the
built-in defaults.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from nwm_region_mgr.parreg.config_schema import AttrDatasetConfig, Config
from nwm_region_mgr.parreg.process_config import (
    ParameterRegionalizationProcessor as PRP,
)
from nwm_region_mgr.utils.config_utils import BaseConfigProcessor

# Column names deliberately different from every built-in default.
CUSTOM_GAGE_COL = "basin_id"
CUSTOM_DIV_COL = "divide_id"
CUSTOM_VPU_COL = "vpu_code"


def _minimal_config_kwargs(**general_overrides):
    """Minimal kwargs for parreg.config_schema.Config, with everything else disabled/empty."""
    general = dict(
        base_dir="/fake/base/dir",
        static_data_dir="/fake/static/dir",
        ngen_hydrofabric_file="vpu_01.gpkg",
        attr_dataset_list=[],
        algorithm_list=[],
    )
    general.update(general_overrides)
    return dict(
        general=general,
        snow_cover={"consider_snowness": False},
        output={},
    )


# -------------------------------renamed id_col, through the real processor properties-------------------------------
@pytest.mark.integration
def test_pipeline_smoke_with_renamed_id_columns(monkeypatch):
    """get_initial_donor_df must use the configured id_col, not the default column names."""
    donors_df = pd.DataFrame(
        {
            CUSTOM_GAGE_COL: ["G1", "G2"],
            "longitude": [-105.0, -104.5],
            "latitude": [40.0, 40.5],
        }
    )
    gage_crosswalk_df = pd.DataFrame(
        {
            CUSTOM_GAGE_COL: ["G1", "G2"],
            CUSTOM_DIV_COL: ["D1", "D2"],
            CUSTOM_VPU_COL: ["01", "01"],
        }
    )

    fake_config = SimpleNamespace(
        general=SimpleNamespace(
            base_dir="/fake/base/dir",
            donor_gage_file="donor_gages.csv",
            gage_divide_cwt_file="gage_divide_crosswalk.parquet",
            id_col=SimpleNamespace(
                gage=CUSTOM_GAGE_COL, divide=CUSTOM_DIV_COL, vpu=CUSTOM_VPU_COL
            ),
        )
    )

    proc = PRP.__new__(PRP)
    proc.config = fake_config

    monkeypatch.setattr(PRP, "donors_df", property(lambda self: donors_df))
    monkeypatch.setattr(PRP, "donor_gages", property(lambda self: donors_df))
    monkeypatch.setattr(PRP, "gage_crosswalk", property(lambda self: gage_crosswalk_df))
    monkeypatch.setattr(PRP, "donor_basins_all", property(lambda self: ["G1", "G2"]))

    donors = proc.donors_df
    donor_df = proc.get_initial_donor_df("01")

    assert not donors.empty
    assert set(donor_df.columns) >= {CUSTOM_DIV_COL, CUSTOM_GAGE_COL}
    assert len(donor_df) == 2
    assert donor_df[CUSTOM_GAGE_COL].tolist() == ["G1", "G2"]


# -------------------------------open attribute-dataset registry (real Config parsing)-------------------------------
@pytest.mark.unit
def test_config_accepts_fully_custom_attr_dataset_name(tmp_path):
    """general.attr_dataset_list/attr_datasets accept an arbitrary name, not just the 4 built-ins."""
    attrs_file = tmp_path / "attrs.parquet"
    pd.DataFrame({"div_id": ["a", "b"], "foo": [1, 2]}).to_parquet(attrs_file)

    config = Config(
        **_minimal_config_kwargs(attr_dataset_list=["my_custom_hydrofabric_attrs"]),
        attr_datasets={
            "my_custom_hydrofabric_attrs": {
                "attr_list": ["foo"],
                "attr_data_file": str(attrs_file),
            }
        },
    )

    assert list(config.attr_datasets.keys()) == ["my_custom_hydrofabric_attrs"]


@pytest.mark.unit
def test_config_rejects_referenced_but_undefined_attr_dataset(tmp_path):
    """A dataset listed in attr_dataset_list but missing from attr_datasets is a clear error."""
    with pytest.raises(ValidationError, match="missing from the 'attr_datasets' section"):
        Config(**_minimal_config_kwargs(attr_dataset_list=["does_not_exist"]))


# -------------------------------eager attribute-dataset validation (fail fast)-------------------------------
@pytest.mark.unit
def test_attr_dataset_validate_schema_happy_path(tmp_path):
    """validate_schema resolves attr_list and checks it against attr_data_file's schema."""
    pq_path = tmp_path / "attrs.parquet"
    pd.DataFrame({"div_id": ["a", "b"], "foo": [1, 2]}).to_parquet(pq_path)

    dataset = AttrDatasetConfig(attr_list=["foo"], attr_data_file=str(pq_path))
    dataset.validate_schema()  # should not raise


@pytest.mark.unit
def test_attr_dataset_validate_schema_missing_file_raises(tmp_path):
    """A missing attr_data_file is caught without reading any data."""
    dataset = AttrDatasetConfig(
        attr_list=["foo"], attr_data_file=str(tmp_path / "missing.parquet")
    )
    with pytest.raises(FileNotFoundError):
        dataset.validate_schema()


@pytest.mark.unit
def test_attr_dataset_validate_schema_all_attrs_missing_raises(tmp_path):
    """If none of attr_list is present in attr_data_file, that's a hard error."""
    pq_path = tmp_path / "attrs.parquet"
    pd.DataFrame({"div_id": ["a", "b"], "foo": [1, 2]}).to_parquet(pq_path)

    dataset = AttrDatasetConfig(attr_list=["nope"], attr_data_file=str(pq_path))
    with pytest.raises(ValueError, match="No valid attributes found"):
        dataset.validate_schema()


@pytest.mark.integration
def test_processor_validate_files_validates_attr_datasets(tmp_path, monkeypatch):
    """ParameterRegionalizationProcessor.validate_files eagerly validates attr_datasets.

    Wraps BaseConfigProcessor.validate_files as a no-op so this isolates the new attr_datasets
    loop from the (unrelated) file-path existence checks for the other config sections.
    """
    pq_path = tmp_path / "attrs.parquet"
    pd.DataFrame({"div_id": ["a", "b"], "foo": [1, 2]}).to_parquet(pq_path)

    config = Config(
        **_minimal_config_kwargs(attr_dataset_list=["custom"]),
        attr_datasets={"custom": {"attr_list": ["foo"], "attr_data_file": str(pq_path)}},
    )

    proc = PRP.__new__(PRP)
    proc.config = config

    monkeypatch.setattr(BaseConfigProcessor, "validate_files", lambda self: None)
    proc.validate_files()  # should not raise

    config.attr_datasets["custom"].attr_data_file = str(tmp_path / "missing.parquet")
    with pytest.raises(ValueError, match="Attribute dataset 'custom'"):
        proc.validate_files()
