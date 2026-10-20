"""Configuration schema for the NGEN simulation application.

config_schema.py defines the Pydantic models that represent the configuration

Classes:
    - Config: Top-level configuration for NGEN simulation.

"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

from nwm_region_mgr.utils import (
    BaseConfig,
    BaseGeneralConfig,
    BaseOutputConfig,
)

logger = logging.getLogger(__name__)

TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S"
TIMESTAMP_FMT1 = "%Y-%m-%d %H:%M:%S"

# Built-in mapping of domain name -> the ngen-forcing pipeline's domain name, used to build
# the MSWM config template. Override or extend via NgenGeneralSettings.ngen_domain_map for a
# domain not already known here (e.g., one added via general.valid_vpus).
DEFAULT_NGEN_DOMAIN_MAP: Dict[str, str] = {
    "conus": "CONUS",
    "ak": "Alaska",
    "hi": "Hawaii",
    "prvi": "Puerto_Rico",
    "gl": "gl",
}

# Built-in mapping of domain name -> NGEN forcing source. A domain not listed here defaults
# to 'nwm' (matching this package's long-standing default), so only 'conus' needs an entry.
DEFAULT_FORCING_SOURCE_MAP: Dict[str, str] = {
    "conus": "aorc",
}


def validate_timestamp(value: str) -> str:
    """Validate that the given string matches one of the supported timestamp formats."""
    for fmt in (TIMESTAMP_FMT, TIMESTAMP_FMT1):
        try:
            datetime.strptime(value, fmt)
            return value
        except ValueError:
            continue

    raise ValueError(
        f"Invalid timestamp '{value}'. Expected format "
        f"{TIMESTAMP_FMT!r} or {TIMESTAMP_FMT1!r}"
    )


class NgenGeneralSettings(BaseGeneralConfig):
    """General settings for the NGEN simulation application."""

    start_time: str | Path = Field(
        description="Start time for the NGEN simulation in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).",
        examples="2022-10-01T00:00:00",
        default="2022-10-01T00:00:00",
    )
    end_time: str | Path = Field(
        description="End time for the NGEN simulation in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).",
        examples="2022-10-01T10:00:00",
        default="2022-10-02T00:00:00",
    )

    algorithm_list: List[
        Literal["gower", "urf", "kmeans", "kmedoids", "hdbscan", "birch", "proximity"]
    ] = Field(
        description="Algorithms to use. Valid options ('gower', 'urf', 'kmeans', 'kmedoids', 'hdbscan', 'birch', 'proximity').",
        examples=["gower", "kmeans"],
        default=["gower"],
    )

    par_file: Path | str | Dict[str, Path] | Dict[str, str] = Field(
        description="Path to the formulation parameters file for NGEN simulation.",
        examples="outputs/region/{run_name}/params/formulation_params_{algorithm_list}_conus_vpu{vpu}.csv",
    )
    pair_file: Path | str | Dict[str, Path] | Dict[str, str] = Field(
        description="Path to the pairing file for NGEN simulation.",
        examples="outputs/region/{run_name}/pairs/pairs_{algorithm_list}_conus_vpu{vpu}_mswm.csv",
    )

    config_template: Path | str = Field(
        default="default_mswm_template.txt",
        description="Path to the MSWM configuration template file for NGEN simulation.",
        examples=["default_mswm_template.txt"],
    )

    ngen_domain_map: Dict[str, str] | None = Field(
        description=(
            "Optional extension of the mapping from domain name to the NGEN-forcing "
            "pipeline's domain name (DEFAULT_NGEN_DOMAIN_MAP), used to build the MSWM "
            "config template. For a domain name also present in the built-in mapping, the "
            "value here replaces it; a new domain name adds an entry. Set this to run an "
            "NGEN simulation for a domain not already known to this package (e.g., one "
            "added via general.valid_vpus)."
        ),
        examples={"gl": "Great_Lakes"},
        default=None,
    )

    forcing_source_map: Dict[str, str] | None = Field(
        description=(
            "Optional extension of the mapping from domain name to NGEN forcing source "
            "('aorc' or 'nwm'). Domains not listed here (directly or via "
            "DEFAULT_FORCING_SOURCE_MAP) default to 'nwm'."
        ),
        examples={"gl": "nwm"},
        default=None,
    )

    # validate timestamp fields
    _validate_start = field_validator("start_time")(validate_timestamp)
    _validate_end = field_validator("end_time")(validate_timestamp)


class NgenOutputConfig(BaseModel):
    """Output configuration for formulation regionalization."""

    ngen: BaseOutputConfig = Field(
        description=(
            "Root directory for the NGEN simulation inputs and outputs, under which sub-directories "
            "'regionalization/{run_name}_{algorithm}/vpu_{vpu}/' will be created for each algorithm and VPU.",
        ),
        examples={
            "save": True,
            "path": "{base_dir}/outputs/ngen",
        },
    )

    config_final: BaseOutputConfig = Field(
        description=(
            "Output configuration for the final configuration file after processing, with placeholders resolved."
        ),
        examples={
            "save": True,
            "path": "{base_dir}/outputs/{run_name}/config_ngen_final.yaml",
        },
    )


class Config(BaseConfig):
    """Top-level configuration for formulation regionalization."""

    general: NgenGeneralSettings = Field(
        description="General settings for NGEN simulation",
        default_factory=NgenGeneralSettings,
    )

    output: NgenOutputConfig = Field(
        description="Output configuration for NGEN simulation.",
        default_factory=NgenOutputConfig,
    )
