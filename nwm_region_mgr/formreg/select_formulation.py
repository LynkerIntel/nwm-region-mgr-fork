"""Select formulations for regionalization.

This module provides functions to select formulations based on summary scores of calibrated basins,
as well as formulation costs if provided in the configuration.

Functions:
    - _validate_nested_huc_ids: Validate that the configured 'huc12' id column holds nested
      numeric HUC codes before relying on string-prefix truncation to find coarser spatial units.
    - _normalize_numeric_huc_artifacts: Repair numeric-dtype artifacts (lost leading zeros,
      trailing '.0') in the 'huc12' id column before the above validation.
    - _compute_huc_id_column: Derive the spatial-unit ID at a given digit width from the
      'huc12' id column, without truncating a hydrofabric's own native/finest-level ID.
    - _get_gages_with_shared_formulations: Identify the formulations shared by a number of gages.
    - _find_gages_nearest_neighbor: Find gages for a given HUC ID using the nearest neighbor method.
    - _find_gages_upscaling: Find gages for a given HUC ID using the upscaling method.
    - _find_calibration_gages: Find gages for a given HUC ID in the summary scores DataFrame.
    - _get_formulation_costs: Retrieve formulation costs from the configuration.
    - _select_formulation_given_score: Select a formulation for each spatial unit based on the method specified.
    - _identify_best_formulation_per_gage: Identify the best formulation for each gage based on summary scores and costs.
    - select_formulation_donors_only: Select formulations for donor basins only.
    - select_formulation_all: Select formulations for all basins.
    - save_formulation_results: Save the formulation selection results to file.
    - plot_formulation_results: Plot the formulation selection results.
    - select_formulation: Head function to select formulations for each spatial unit in the VPU.

"""

import logging
import re
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd

from nwm_region_mgr.formreg import config_schema as cs
from nwm_region_mgr.utils import (
    check_columns_dataframe,
    find_gages_within_buffer,
    read_table,
)

logger = logging.getLogger(__name__)

# True USGS Watershed Boundary Dataset HUC12 codes are 12-digit numeric strings, where a
# coarser unit's ID (e.g. HUC8) is the leading-digit prefix of its children's HUC12 IDs. The
# "upscaling" basin_fill_method, and any 'huc_level' coarser than the finest/native level,
# rely on that nesting to find a parent spatial unit by truncating the ID string.
_NESTED_HUC_ID_PATTERN = re.compile(r"^\d{12}$")


def _validate_nested_huc_ids(huc12_values: pd.Series, huc12_col: str) -> None:
    """Validate that `huc12_col` values look like nested, 12-digit numeric HUC codes.

    Called only when truncating the configured 'huc12' id column to a coarser spatial unit
    (i.e., spatial_unit.basin_fill_method == 'upscaling', or 'huc_level' finer than huc12).
    If the hydrofabric's id_col.huc12 column doesn't follow the nested numeric-prefix
    convention, truncating it produces spatial groupings with no real geographic meaning.
    Raising here surfaces that mismatch immediately instead of silently mis-grouping basins.

    Args:
        huc12_values: Values of the configured 'huc12' id column.
        huc12_col: Name of that column, for the error message.

    Raises:
        ValueError: If any non-null value does not match a 12-digit numeric string.

    """
    values = huc12_values.dropna().astype(str)
    if values.empty:
        return

    non_conforming = values[~values.str.match(_NESTED_HUC_ID_PATTERN)]
    if not non_conforming.empty:
        examples = sorted(non_conforming.unique())[:5]
        raise ValueError(
            f"spatial_unit.basin_fill_method='upscaling' (or a 'huc_level' coarser than the "
            f"hydrofabric's native/finest spatial unit) requires the '{huc12_col}' column "
            f"(general.id_col.huc12) to contain nested 12-digit numeric HUC codes, following "
            "the USGS Watershed Boundary Dataset convention where a coarser unit's ID is the "
            f"digit-prefix of its children's IDs. Found values that don't match, e.g.: "
            f"{examples}. If this hydrofabric does not use nested HUC-style IDs, map "
            "general.id_col.huc12 to the hydrofabric's own atomic spatial-unit ID column, set "
            "spatial_unit.huc_level to that same (finest) level so no truncation is needed, "
            "and set spatial_unit.basin_fill_method='nearest-neighbor', which does not require "
            "nested IDs."
        )


def _normalize_numeric_huc_artifacts(huc12_values: pd.Series) -> pd.Series:
    """Repair common numeric-dtype artifacts in a 'huc12' id column before HUC validation.

    A HUC12 code is sometimes stored upstream (e.g. in a parquet file) as a numeric dtype
    rather than string. Read back as text, that silently drops leading zeros (int -> e.g.
    '10100020101', 11 digits instead of 12) or appends a trailing '.0' (float -> e.g.
    '10100020101.0'). Left uncorrected, a genuinely nested HUC12 column would be misdiagnosed
    by _validate_nested_huc_ids as not following the HUC12 convention, purely because of its
    source dtype. Strips a trailing '.0', and zero-pads a purely numeric value shorter than
    12 digits back to 12 digits; any value that isn't purely numeric (i.e., not this specific
    artifact) is left untouched, as are null entries.

    Args:
        huc12_values: Values of the configured 'huc12' id column.

    Returns:
        A new Series with the same index, numeric-dtype artifacts repaired.

    """
    not_null = huc12_values.notna()
    values = huc12_values.astype(str).str.replace(r"\.0$", "", regex=True)

    needs_padding = (
        not_null & values.str.fullmatch(r"\d+").fillna(False) & (values.str.len() < 12)
    )
    values = values.where(~needs_padding, values.str.zfill(12))

    return values.where(not_null, huc12_values)


def _compute_huc_id_column(huc12_values: pd.Series, huc_digit: int) -> pd.Series:
    """Derive the spatial-unit ID at `huc_digit` digits from the configured 'huc12' id column.

    huc_digit >= 12 (the finest configurable level) is a true no-op: the hydrofabric's native
    'huc12' id is used as-is, not truncated to exactly 12 characters. Truncating it would
    silently collide two different divides that happen to share the same 12-character prefix,
    for a hydrofabric whose native ID is longer than 12 characters (e.g., a custom hydrofabric
    using the huc_level='huc12' + basin_fill_method='nearest-neighbor' escape hatch described
    on FormulationSpatialUnitConfig.huc_level).

    Args:
        huc12_values: Values of the configured 'huc12' id column.
        huc_digit: Number of leading characters to keep (derived from spatial_unit.huc_level).

    Returns:
        A new Series of the resulting spatial-unit IDs, as strings.

    """
    if huc_digit >= 12:
        return huc12_values.astype(str)
    return huc12_values.str[:huc_digit]


def _get_gages_with_shared_formulations(
    df: pd.DataFrame,
    min_gages: int = 3,
    gage_col: str = "gage_id",
) -> Tuple[bool, List[str], List[str]]:
    """Identify the largest formulation set shared by at least `min_gages` gages.

    Args:
        df (pd.DataFrame): Must contain `gage_col` and 'formulation' columns.
        min_gages (int): Minimum number of gages that must share the same formulations.
        gage_col (str): Column name for gage ID in `df`.

    Returns:
        (True, [gages], [shared_formulations]) if found, else (False, [], [])

    """
    # Build mapping from gage ID -> set of formulations
    gage_to_formulations = df.groupby(gage_col)["formulation"].apply(
        lambda x: frozenset(x)
    )

    # All unique gage_ids and their formulation sets
    gage_ids = list(gage_to_formulations.index)
    gage_form_sets = gage_to_formulations.tolist()

    # Step 1: Get all unique formulations across gages
    all_formulations = set().union(*gage_form_sets)

    # Step 2: Loop through combinations of formulations, largest sets first
    for r in range(len(all_formulations), 0, -1):  # Start from largest combos
        for combo in combinations(all_formulations, r):
            combo_set = set(combo)
            matching_gages = [
                gage_ids[i]
                for i, g_form in enumerate(gage_form_sets)
                if combo_set.issubset(g_form)
            ]
            if len(matching_gages) >= min_gages:
                logger.debug(
                    f"Found {len(matching_gages)} gages ({matching_gages}) sharing formulations: {combo_set}."
                )
                return True, matching_gages, list(combo_set)

    return False, [], []


def _find_gages_nearest_neighbor(
    huc_id: str,
    df: pd.DataFrame,
    gage_file: Path,
    gage_col: str,
    min_gages: int,
    gdf: gpd.GeoDataFrame,
) -> tuple[list, list, list]:
    """Find gages for the given HUC ID in the DataFrame.

    Args:
        huc_id (str): HUC ID to find gages for.
        df (pd.DataFrame): DataFrame of summary scores.
        gage_file (Path): Path to the donor gage file containing gage_id, longitude, and latitude.
        gage_col (str): Column name for gage ID in the gage file.
        min_gages (int): Minimum number of gages required for each spatial unit.
        gdf (gpd.GeoDataFrame): GeoDataFrame of hydrofabric (HUC12) polygons.

    Returns:
        tuple: A tuple containing a list of calibration gage IDs found for the HUC ID,  distances of the gages
            to the HUC centroid, and the shared formulations.

    """
    # get all calibration gages with valid formulation and summary scores
    gages_calib = (
        df[~df["formulation"].isna() & ~df["summary_score"].isna()][gage_col]
        .unique()
        .tolist()
    )

    # iteratively find the nearest neighbor gages for the given HUC ID, increasing the neighborhood size
    # by 100km each time until the minimum number of gages is met
    buffer = 100  # initial buffer size in kilometers
    gages = []
    max_buffer = 2000
    while len(gages) < min_gages and buffer <= max_buffer:
        # find gages within the buffer around huc_id
        gages, distances, _ = find_gages_within_buffer(
            gage_file, gage_col, buffer, gdf=gdf, id=huc_id
        )

        # filter gages to only those that are calibrated, and filter distances accordingly
        filtered = [(g, d) for g, d in zip(gages, distances) if g in gages_calib]
        if filtered:
            gages, distances = map(list, zip(*filtered))
        else:
            gages, distances = [], []

        # if there are enough gages, check if they share the same formulations
        found_gages = False
        if len(gages) >= min_gages:
            found_gages, gages, formulations = _get_gages_with_shared_formulations(
                df[df[gage_col].isin(gages)], min_gages=min_gages, gage_col=gage_col
            )

        # if no gages are found, increase the buffer size and try again
        if not found_gages:
            logger.debug(
                f"{huc_id}: Not enough gages found within {buffer} km buffer. Increasing buffer size by 100km."
            )
            buffer += 100  # increase buffer size by 100 km
        else:
            logger.debug(
                f"{huc_id}: Found {len(gages)} gages ({gages}) within {buffer} km buffer."
            )
            break

    else:
        # if while loop exits normally (len(gages) >= min_gages or buffer > max_buffer)
        if len(gages) < min_gages:
            logger.warning(
                f"Not enough calibrated gages found for {huc_id} after increasing buffer "
                f"to {buffer - 100} km. Minimum required is {min_gages}."
            )

    return gages, distances, formulations


def _find_gages_upscaling(
    huc_id: str, df: pd.DataFrame, gage_col: str, min_gages: int
) -> tuple[list, str, list]:
    """Find gages for the given HUC ID in the DataFrame.

    Args:
        huc_id (str): HUC ID to find gages for.
        df (pd.DataFrame): DataFrame of summary scores.
        gage_col (str): Column name for gage ID in the gage file.
        min_gages (int): Minimum number of gages required for each spatial unit.

    Returns:
        tuple: A tuple containing a list of calibration gage IDs found for the HUC ID, the HUC level,
        and the shared formulations.

    """
    # define valid HUC levels
    valid_huc_levels = ["huc2", "huc4", "huc6", "huc8", "huc10", "huc12"]
    valid_huc_levels.reverse()

    # make sure current huc level is in the valid HUC levels
    current_huc_level = "huc" + str(len(huc_id))
    if current_huc_level not in valid_huc_levels:
        msg = f"Invalid current HUC level: {current_huc_level}. Valid levels are: {', '.join(valid_huc_levels)}."
        logger.error(msg)
        raise ValueError(msg)

    gages = []
    formulations = []
    huc_level = current_huc_level  # Default to current if nothing better is found

    # start from current huc level, loop through valid HUC levels to ensure sufficient calibrated gages
    for _, huc_level in enumerate(
        valid_huc_levels[valid_huc_levels.index(current_huc_level) :]
    ):
        # count the number of unique gages in the new HUC level
        huc_digit = int(huc_level.replace("huc", ""))
        gages = (
            df[df["huc_id"].str[:huc_digit] == huc_id[:huc_digit]][gage_col]
            .unique()
            .tolist()
        )

        # remove NaN values from gages
        gages = [gage for gage in gages if pd.notna(gage)]
        ngage = len(gages)

        # if there are enough gages, check if they share the same formulations
        if ngage >= min_gages:
            found_gages, gages, formulations = _get_gages_with_shared_formulations(
                df[df[gage_col].isin(gages)], min_gages=min_gages, gage_col=gage_col
            )

            if not found_gages:
                logger.debug(
                    f"{huc_id}: upscaling HUC level from {current_huc_level} to {huc_level} "
                )
                continue
            else:
                break

    return gages, huc_level, formulations


def _find_calibration_gages(
    huc_id: str, df: pd.DataFrame, config: cs.Config, gdf: gpd.GeoDataFrame
) -> tuple[str, list, list, list]:
    """Find gages for the given HUC ID in the DataFrame.

    Args:
        huc_id (str): HUC ID to find gages for.
        df (pd.DataFrame): DataFrame of summary scores.
        config (cs.Config): Configuration object containing settings.
        gdf (gpd.GeoDataFrame): GeoDataFrame of hydrofabric (HUC12) polygons.

    Returns:
        tuple: A tuple containing the HUC level, the list of calibration gage IDs found for the HUC ID,
            list of corresponding distances, and the list of shared formulations.

    """
    # get configuration settings
    gage_col = getattr(
        config.general.id_col, "gage", "gage_id"
    )  # column name for gage ID in the DataFrame
    nmin_gages = (
        config.spatial_unit.nmin_calib_basin
    )  #  minimum number of calibration gages required
    method = (
        config.spatial_unit.basin_fill_method.lower()
    )  # method to find additional calibration gages if needed
    gage_file = config.general.donor_gage_file  # path to the donor gage file

    # check if the huc_id is valid
    if not isinstance(huc_id, str) or len(huc_id) < 2:
        msg = f"Invalid HUC ID: {huc_id}. It should be a string with at least 2 characters."
        logger.error(msg)
        raise ValueError(msg)

    # remove rows with invalid gage_col, formulation, and summary_score columns in the DataFrame
    df = df[
        ~df[gage_col].isna() & ~df["formulation"].isna() & ~df["summary_score"].isna()
    ]

    # check number of calibration gages within the current huc_id
    gages0 = df[df["huc_id"] == huc_id][gage_col].unique().tolist()
    gages0 = [gage for gage in gages0 if pd.notna(gage)]  # remove NaN values

    if len(gages0) >= nmin_gages:
        found_gages, gages0, formulations = _get_gages_with_shared_formulations(
            df[df[gage_col].isin(gages0)], min_gages=nmin_gages, gage_col=gage_col
        )

    # if there are enough gages at the current huc_id level, return them
    if len(gages0) >= nmin_gages:
        logger.debug(
            f"{huc_id}: found {len(gages0)} calibration gages ({gages0}) at the current level. No need to find additional gages."
        )
        huc_level = "huc" + str(len(huc_id))
        return huc_level, gages0, [np.nan] * len(gages0), formulations
    else:
        logger.debug(
            f"{huc_id}: found only {len(gages0)} calibration gages ({gages0}) at the current level. "
            f"Need to find additional gages using the {method} method."
        )

    def nearest_neighbor_fallback():
        logger.warning(
            f"Not enough calibration gages found for {huc_id} after upscaling. "
            f"Found {len(gages0)} gages ({gages0}), minimum required is {nmin_gages}. "
            f"Use nearest-neighbor method for {huc_id} instead."
        )
        gages_nn, dists_nn, formulations = _find_gages_nearest_neighbor(
            huc_id, df, gage_file, gage_col, nmin_gages, gdf
        )
        return gages_nn, dists_nn, "huc" + str(len(huc_id)), formulations

    if method == "upscaling":
        # find calibration gages using upscaling method
        gages, huc_level, formulations = _find_gages_upscaling(
            huc_id, df, gage_col, nmin_gages
        )
        dists = [np.nan] * len(gages)
        if len(gages) < nmin_gages:
            gages, dists, huc_level, formulations = nearest_neighbor_fallback()

    elif method == "nearest-neighbor":
        # find calibration gages using nearest neighbor
        gages, dists, formulations = _find_gages_nearest_neighbor(
            huc_id, df, gage_file, gage_col, nmin_gages, gdf
        )
        huc_level = "huc" + str(
            len(huc_id)
        )  # HUC level is determined by the length of huc_id
    else:
        # raise error if the basin fill method is not recognized
        msg = f"Unknown basin fill method: {method}. Supported methods are 'upscaling' and 'nearest-neighbor'."
        logger.error(msg)
        raise ValueError(msg)

    return huc_level, gages, dists, formulations


def _get_formulation_costs(
    config: cs.FormulationCostConfig,
) -> Optional[dict[str, float]]:
    """Get formulation costs from the configuration.

    Args:
        config : cs.FormulationCostConfig
            Configuration object containing formulation cost settings.

    Returns:
        Optional[dict[str, float]]
            Dictionary of formulation costs, keyed by formulation name. If no costs are defined, returns None.

    """
    if config.file:
        # check "formulation" and "cost" columns in the cost file
        check_columns_dataframe(
            config.file,
            columns=["formulation", "cost"],
        )
        # read the cost file
        cost_df = read_table(config.file)
        if not cost_df.empty:
            return dict(zip(cost_df["formulation"], cost_df["cost"]))
        else:
            logger.warning(
                f"No costs found in the file: {config.file}. Returning None."
            )
            return None
    elif config.costs:
        return config.costs
    else:
        logger.warning(
            "No formulation costs defined in the configuration. Returning None."
        )

    return None


def _select_formulation_given_score(
    df: pd.DataFrame,
    method: str = "average_score",
    score_type: str = "basin",
    gage_col: str = "gage_id",
    div_col: str = "div_id",
) -> pd.DataFrame:
    """Compute total or average score for each spatial unit based on the method specified.

    Args:
        df : pd.DataFrame
            DataFrame containing summary scores for each formulation and calibrated basin.
        method : str, optional
            Method to compute total or average score, either 'total_score' or 'average_score', by default 'average_score'.
        score_type : str, optional
            Type of score to compute, either 'basin' or 'divide', by default 'basin'.
        gage_col : str, optional
            Column name for gage ID in the DataFrame, by default "gage_id".
        div_col : str, optional
            Column name for divide ID in the DataFrame, by default "div_id".

    Returns:
        pd.DataFrame
            DataFrame with total scores computed for each spatial unit.

    """
    # determine the column name for the type of subdivision to compute total/average score
    if score_type == "basin" and gage_col in df.columns:
        col1 = gage_col
    elif score_type == "divide" and div_col in df.columns:
        col1 = div_col
    else:
        msg = (
            f"Invalid best_formulation type '{score_type}' or missing required columns in the DataFrame. "
            f"Expected '{gage_col}' for type='basin' or '{div_col}' for type='divide'."
        )
        logger.error(msg)
        raise ValueError(msg)

    df1 = df[[col1, "formulation", "summary_score", "cost"]].copy()

    # remove duplicate rows
    df1 = df1.drop_duplicates(subset=[col1, "formulation"])

    # drop rows with NaN values in the summary_score column
    df1 = df1.dropna(subset=["summary_score"])

    # identify best formulation(s)
    if method == "total_score":
        # Sum of summary_score per formulation
        df1.loc[:, method] = (
            df1.groupby(["formulation"])["summary_score"].transform("sum").round(2)
        )

    elif method == "average_score":
        # Average summary_score per formulation
        df1.loc[:, method] = (
            df1.groupby(["formulation"])["summary_score"].transform("mean").round(2)
        )

    else:
        msg = f"Unknown method: {method}. Supported methods are 'total_score' and 'average_score'."
        logger.error(msg)
        raise ValueError(msg)

    # keep only the best formulation(s) with the highest total or average score
    df1 = df1[df1[method] == df1[method].max()]

    # if there are multiple formulations with the same score, choose the one that has the lowest cost
    if df1.shape[0] > 1:
        if df1["cost"].notna().any():
            df1 = df1.loc[[df1["cost"].idxmin()]]
        else:
            df1 = df1.iloc[[0]]

    # drop col1
    df1 = df1.drop(columns=[col1])

    # remove duplicates
    df1 = df1.drop_duplicates(subset=["formulation", method])

    return df1


def _identify_best_formulation_per_gage(
    df_score: pd.DataFrame,
    gage_col: str,
    tolerance: float = 0.05,
    cost_dict: Optional[dict] = None,
) -> pd.DataFrame:
    """Identify the best formulation for each gage based on summary scores and costs.

    Args:
        df_score : pd.DataFrame
            DataFrame containing summary scores for each formulation and calibrated basin.
        gage_col : str
            Column name for gage ID in the gage file.
        tolerance : float, optional
            Tolerance for the summary score to consider formulations as equally good, by default 0.05.
        cost_dict : dict, optional
            Dictionary containing costs for each formulation, by default None.

    Returns:
        pd.DataFrame
            DataFrame with the best formulation for each gage and its score.

    """
    # for each gage, find the best formulation(s) within the tolerance
    max_scores = df_score.groupby(gage_col)["summary_score"].transform("max")
    best_per_gage = df_score[
        df_score["summary_score"] >= (max_scores - max_scores * tolerance)
    ].copy()

    # if formulation costs are provided, select the formulation with the minimum cost; otherwise,
    # choose the formulation with the highest score (this essentially ignores the tolerance)
    if cost_dict is not None:
        # check if all formulations are in the cost_dict
        if not best_per_gage["formulation"].isin(cost_dict.keys()).all():
            missed_formulations = best_per_gage[
                ~best_per_gage["formulation"].isin(cost_dict.keys())
            ]["formulation"].unique()

            msg = f"Some formulations are missing costs in the configuration: {', '.join(missed_formulations)}."
            logger.error(msg)
            raise ValueError(msg)

        # map costs to the formulations
        best_per_gage.loc[:, "cost"] = best_per_gage["formulation"].map(cost_dict)

        # for each gage, select the formulation with the minimum cost
        best_per_gage = best_per_gage.loc[
            best_per_gage.groupby(gage_col)["cost"].idxmin()
        ].copy()

        # if there are multiple formulations with the same minimum cost, keep the first one
        best_per_gage = best_per_gage.drop_duplicates(subset=[gage_col])
    else:
        # choose the first formulation with the highest score for each gage
        best_per_gage = best_per_gage.loc[
            best_per_gage.groupby(gage_col)["summary_score"].idxmax()
        ].copy()

        # initialize cost column to None
        best_per_gage["cost"] = None

    return best_per_gage


def select_formulation_donors_only(
    config: cs.Config,
    vpu: str,
    df_score: pd.DataFrame,
    cost_dict: Optional[dict] = None,
) -> pd.DataFrame:
    """Select formulations for donor basins only based on summary scores and costs.

    Args:
        config : cs.Config
            Configuration object containing settings for the regionalization.
        vpu : str
            Vector Processing Unit (VPU) for which to select formulations.
        df_score : pd.DataFrame
            DataFrame containing summary scores for each formulation and calibrated basin.
        cost_dict : Optional[dict], optional
            Dictionary containing costs for each formulation, by default None.

    Returns:
        pd.DataFrame
            DataFrame with selected formulations and their scores.

    """
    # get the ID columns from the configuration
    gage_col = getattr(
        config.general.id_col, "gage", "gage_id"
    )  # column name for gage ID in the DataFrame
    div_col = getattr(config.general.id_col, "divide", "div_id")
    vpu_col = getattr(config.general.id_col, "vpu", "vpu_id")

    # score computing method, type, and tolerance
    score_tolerance = config.spatial_unit.best_formulation.tolerance

    # identify the best formulation for each gage
    df_best_per_gage = _identify_best_formulation_per_gage(
        df_score,
        gage_col=gage_col,
        tolerance=score_tolerance,
        cost_dict=cost_dict,
    )

    # read the crosswalk file for gage/divide relationships
    col_dtype = {
        gage_col: "string",
        div_col: "string",
        vpu_col: "string",
    }
    cwt_divide_gage = read_table(config.general.gage_divide_cwt_file, dtype=col_dtype)

    # filter by vpu
    cwt_divide_gage = cwt_divide_gage[cwt_divide_gage[vpu_col] == vpu].copy()

    # merge the crosswalk with the best formulations DataFrame
    if div_col in df_best_per_gage.columns:
        df_best_per_gage = df_best_per_gage.drop(columns=[div_col])
    df_selected = df_best_per_gage.merge(
        cwt_divide_gage[[gage_col, div_col]].drop_duplicates(),
        on=gage_col,
        how="inner",
    )

    return df_selected


def select_formulation_all(
    config: cs.Config,
    vpu: str,
    df_score: pd.DataFrame,
    gdf_vpu: gpd.GeoDataFrame,
    cost_dict: Optional[dict] = None,
) -> pd.DataFrame:
    """Select formulation for all catchments in the VPU.

    Args:
        config : cs.Config
            Configuration object containing settings for the regionalization.
        vpu : str
            Vector Processing Unit (VPU) for which to select formulations.
        df_score : pd.DataFrame
            DataFrame containing summary scores for each formulation and calibrated basin.
        gdf_vpu : gpd.GeoDataFrame
            GeoDataFrame of the VPU polygons, used for spatial operations where needed.
        cost_dict : Optional[dict], optional
            Dictionary containing costs for each formulation, by default None.

    Returns:
        pd.DataFrame
            DataFrame with selected formulations and their scores.

    """
    # get the ID columns from the configuration
    id_cols = config.general.id_col
    gage_col = getattr(id_cols, "gage", "gage_id")
    div_col = getattr(id_cols, "divide", "div_id")
    huc12_col = getattr(id_cols, "huc12", "huc_12")
    huc12_layer = getattr(config.general.layer_name, "huc12", "WBDSnapshot_National")

    # score computing method, type, and tolerance
    score_method = config.spatial_unit.best_formulation.method.lower()
    score_type = config.spatial_unit.best_formulation.type.lower()
    score_tolerance = config.spatial_unit.best_formulation.tolerance
    logger.info(
        f"Computing total score using method '{score_method}' and type '{score_type}'."
    )

    # crosswalk for gage/divide
    col_dtype = {
        gage_col: "string",
        div_col: "string",
        huc12_col: "string",
    }
    cwt_divide_gage = read_table(config.general.gage_divide_cwt_file, dtype=col_dtype)

    # crosswalk for divide/huc12
    cwt_divide_huc12 = read_table(config.general.divide_huc12_cwt_file, dtype=col_dtype)

    # enforce consistency (critical)
    for df in [cwt_divide_gage, cwt_divide_huc12]:
        df[div_col] = df[div_col].astype("string")

    gdf_vpu[div_col] = gdf_vpu[div_col].astype("string")

    # filter cwt_divide_huc12 with divides in gdf_vpu (i.e., only keep divides that are in the current VPU)
    cwt_divide_huc12 = cwt_divide_huc12[
        cwt_divide_huc12[div_col].isin(gdf_vpu[div_col])
    ].copy()

    # get spatial units for formulation selection
    huc_level = config.spatial_unit.huc_level.lower().replace("-", "").replace("_", "")
    huc_digit = int(huc_level.replace("huc", ""))
    basin_fill_method = config.spatial_unit.basin_fill_method.lower()

    # validate the nested-HUC-ID assumption up front, but only when it's actually relied on:
    # 'upscaling' truncates huc12_col at progressively coarser levels regardless of huc_level
    # (see _find_gages_upscaling), and any huc_level finer than the hydrofabric's native/finest
    # spatial unit (huc_digit < 12) truncates it once below.
    if basin_fill_method == "upscaling" or huc_digit < 12:
        # repair common numeric-dtype artifacts (e.g. huc12_col stored as float64/int64
        # upstream) before validating, so a genuinely nested HUC12 column isn't misdiagnosed
        # as non-conforming purely because of its source dtype. Only applied when the nested-
        # HUC interpretation is actually in use, so a custom hydrofabric's genuinely short
        # native numeric IDs (used via the huc_level='huc12' no-truncation case below) are
        # never rewritten.
        cwt_divide_huc12[huc12_col] = _normalize_numeric_huc_artifacts(
            cwt_divide_huc12[huc12_col]
        )
        _validate_nested_huc_ids(cwt_divide_huc12[huc12_col], huc12_col)

    # add huc_id column to cwt_divide_huc12 based on huc_digit (see _compute_huc_id_column)
    cwt_divide_huc12["huc_id"] = _compute_huc_id_column(
        cwt_divide_huc12[huc12_col], huc_digit
    )

    # make sure type of huc_id column is string
    if not pd.api.types.is_string_dtype(cwt_divide_huc12["huc_id"]):
        logger.warning(
            f"Converting 'huc_id' column to string type. Current type: {cwt_divide_huc12['huc_id'].dtype}."
        )
        cwt_divide_huc12["huc_id"] = cwt_divide_huc12["huc_id"].astype(str)

    # merge the two crosswalks first
    df_cwt = cwt_divide_huc12[["huc_id", huc12_col, div_col]].merge(
        cwt_divide_gage[[gage_col, div_col]].drop_duplicates(),
        on=div_col,
        how="left",
    )

    # then merge with the score DataFrame
    df_score = df_score.merge(df_cwt, on=gage_col, how="outer")

    logger.info(
        f"Selecting formulations for VPU {vpu} at {huc_level} level. "
        f"There are {len(df_score['huc_id'].unique())} unique {huc_level} IDs."
    )

    # all unique huc_ids in the score DataFrame
    huc_ids = df_score["huc_id"].unique()

    # remove NaN values from huc_ids
    huc_ids = [huc_id for huc_id in huc_ids if pd.notna(huc_id)]

    # get huc12 geometry for these huc_ids (to compute centroid distances between gages and huc_ids)
    huc12_hydro_file = Path(config.general.huc12_hydrofabric_file).resolve(strict=True)

    # Open with Fiona to read features in huc_ids
    features = []
    huc_digit = len(huc_ids[0])  # assuming all huc_ids have the same length
    with fiona.open(
        huc12_hydro_file, "r", layer=huc12_layer, open_options=["METHOD=ONLY_CCW"]
    ) as src:
        for feat in src:
            huc_key = next(
                (k for k in feat["properties"] if k.lower() == huc12_col), None
            )
            if huc_key and feat["properties"][huc_key][:huc_digit] in huc_ids:
                features.append(feat)

        if not features:
            msg = (
                f"No matching HUC12 geometries found in {huc12_hydro_file} "
                f"for the specified HUC IDs at {huc_level} level."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Convert to GeoDataFrame
        huc12_gdf = gpd.GeoDataFrame.from_features(features, crs=src.crs)

    # loop through each unique huc_id and select formulation
    df_selected = pd.DataFrame()
    for huc_id in huc_ids:
        logger.debug(f"Processing HUC ID: {huc_id}")

        # Find actual column name in gdf that matches huc12_col (case-insensitive)
        col_match = next(
            (col for col in huc12_gdf.columns if col.lower() == huc12_col.lower()),
            None,
        )

        # Use the matched column to filter huc12_gdf
        huc12_gdf_huc = huc12_gdf[
            huc12_gdf[col_match].astype(str).str[:huc_digit] == huc_id
        ].copy()

        if huc12_gdf_huc.empty:
            msg = f"No geometry found for HUC ID: {huc_id}. Please check the HUC12 hydrofabric file."
            logger.error(msg)
            raise ValueError(msg)

        huc_level, gages, dists, formulations = _find_calibration_gages(
            huc_id, df_score, config, huc12_gdf_huc
        )
        if not gages:
            msg = (
                f"No calibration gages found for HUC ID: {huc_id}. "
                f"Minimum required is {config.spatial_unit.nmin_calib_basin}."
            )
            logger.error(msg)
            raise ValueError(msg)

        # filter the score DataFrame for identified gages and formulations
        df_huc = df_score[
            df_score[gage_col].isin(gages) & df_score["formulation"].isin(formulations)
        ].copy()

        # identify best formulation for each gage based on summary scores and/or formulation costs
        df_huc = _identify_best_formulation_per_gage(
            df_huc,
            gage_col=gage_col,
            tolerance=score_tolerance,
            cost_dict=cost_dict,
        )

        # select the best formulation for each huc_id based on the score method and type specified in the configuration
        best_formulation = _select_formulation_given_score(
            df_huc, score_method, score_type, gage_col, div_col
        )

        # append the selected formulation to the list
        if best_formulation.empty:
            logger.warning(
                f"No valid formulations found for HUC8 ID: {huc_id}. Skipping this HUC."
            )
            continue

        # add huc_id, huc_level, number of gages and vpu to the best formulation
        best_formulation["huc_id"] = huc_id
        best_formulation["upscale_huc"] = huc_level
        best_formulation["num_gages"] = len(gages)
        best_formulation["distances"] = ", ".join(
            [str(d) for d in dists]
        )  # join distances as a string

        df_selected = pd.concat([df_selected, best_formulation], ignore_index=True)

    # merge with cwt_divide_huc12 to get the divide_id
    if df_selected.empty:
        logger.warning(
            "No formulations were selected for any HUC IDs. Returning an empty DataFrame."
        )
    else:
        df_selected = df_selected.merge(
            cwt_divide_huc12[["huc_id", div_col]].drop_duplicates(),
            on="huc_id",
            how="left",
        )

    # handle formulation selection for calibration gages
    if config.general.approach_calib_basins == "regionalization":
        pass
    elif config.general.approach_calib_basins == "summary_score":
        # select formulations for donor gages based on the summary scores and costs
        df_selected_donors = select_formulation_donors_only(
            config, vpu, df_score, cost_dict=cost_dict
        )

        if df_selected.empty:
            df_selected = df_selected_donors.copy()
        else:
            # replace the formulation for donors in df_selected with the one from df_selected_donors
            # first ensure unique mapping from donors
            df_donors_unique = df_selected_donors.drop_duplicates(div_col).set_index(
                div_col
            )

            # align df_selected on the same index
            df_selected = df_selected.set_index(div_col)

            # update entire rows where keys match
            df_selected.update(df_donors_unique)

            # restore index
            df_selected = df_selected.reset_index()
    else:
        msg = (
            f"Unknown approach for assigning formulations to calibrated basins: "
            f"{config.general.approach_calib_basins}. Supported methods are 'regionalization' and 'summary_score'."
        )
        logger.error(msg)
        raise ValueError(msg)

    return df_selected


def check_gage_formulation_uniqueness(
    df_form: pd.DataFrame,
    config: cs.Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check that each gage has a unique formulation assigned.

    Args:
        df_form : pd.DataFrame
            DataFrame with selected formulations for each divide.
        config : cs.Config
            Configuration object containing settings for the regionalization.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing the new DataFrame with gage-formulation mapping, and
            the updated DataFrame with divide-formulation mapping and


    """
    # get formulation selected for each donor gage
    gage_col = getattr(config.general.id_col, "gage", "gage_id")
    div_col = getattr(config.general.id_col, "divide", "div_id")
    df_cwt = read_table(
        config.general.gage_divide_cwt_file,
        dtype={gage_col: "str", div_col: "str"},
    )
    df_cwt = df_cwt[[gage_col, div_col]].drop_duplicates()
    df_form_gage = df_form[[div_col, "formulation"]].drop_duplicates()
    df_form_gage = df_cwt.merge(
        df_form_gage,
        on=div_col,
        how="inner",
    )

    # check if any gage is associated with multiple formulations
    df_count = (
        df_form_gage[[gage_col, "formulation"]]
        .drop_duplicates()
        .groupby(gage_col)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )
    gages = df_count[df_count["count"] > 1][gage_col].tolist()

    # loop through gages associated with multiple formulations to reassign them to
    # the most frequent formulation for both df_form_gage and df_form
    for gage in gages:
        df1 = df_form_gage.loc[df_form_gage[gage_col] == gage].sort_values(
            by="formulation"
        )
        df_count = (
            df1.groupby("formulation")
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
        )
        most_freq_form = df_count.iloc[0]["formulation"]
        logger.debug(
            f"Gage {gage} is associated with multiple formulations, "
            f"reassigning to the most frequent formulation {most_freq_form}"
        )
        df_form_gage.loc[df_form_gage[gage_col] == gage, "formulation"] = most_freq_form
        df_form.loc[df_form[div_col].isin(df1[div_col]), "formulation"] = most_freq_form

    # ensure no duplicate rows in df_form_gage
    df_form_gage = df_form_gage[[gage_col, "formulation"]].drop_duplicates()

    return df_form_gage, df_form


def save_formulation_results(
    df_form: pd.DataFrame,
    df_form_gage: pd.DataFrame,
    config: cs.Config,
    vpu: str,
) -> None:
    """Save the selected formulations to the output file.

    Args:
        df_form : pd.DataFrame
            DataFrame with formualtion selected for each divide.
        df_form_gage : pd.DataFrame
            DataFrame with formualtion selected for each gage.
        config : cs.Config
            Configuration object containing settings for the regionalization.
        vpu : str
            Vector Processing Unit (VPU) for which to save formulations.

    """
    co = getattr(config.output, "formulation", None)
    if co is not None:
        co.save_to_file(
            df_form, vpu=vpu, data_str="Formulation Selection", use_stem_suffix=False
        )

    # merge with calibration parameter file to get parameters for each gage
    gage_col = getattr(config.general.id_col, "gage", "gage_id")
    df_pars = read_table(config.general.calib_param_file, dtype={gage_col: "str"})
    df_pars = df_form_gage.merge(
        df_pars, on=[gage_col, "formulation"], how="inner"
    ).drop_duplicates()

    # save the parameters for each gage to the output file
    co.save_to_file(
        df_pars,
        vpu=vpu,
        data_str="Formulation Selection (per gage with parameters)",
        use_stem_suffix=True,
    )


def plot_formulation_results(
    df_selected: pd.DataFrame,
    config: cs.Config,
    vpu: str,
    gdf_vpu: Optional[gpd.GeoDataFrame],
) -> None:
    """Plot the selected formulations for each spatial unit in the VPU.

    Args:
        df_selected : pd.DataFrame
            DataFrame with selected formulations and their scores.
        config : cs.Config
            Configuration object containing settings for the regionalization.
        vpu : str
            Vector Processing Unit (VPU) for which to plot formulations.
        gdf_vpu : Optional[gpd.GeoDataFrame], optional
            GeoDataFrame of the VPU polygons, used for spatial operations where needed

    """
    # generate plots if enabled
    cc = getattr(config.output, "formulation", None)
    if cc is None:
        return

    div_col = getattr(config.general.id_col, "divide", "div_id")
    score_method = config.spatial_unit.best_formulation.method.lower()
    if any(cc.plots.values()):
        # get geometry for divides if spatial map is enabled
        if cc.plots.get("spatial_map", False):
            # merge the geometry with the selected formulations
            df_selected = df_selected.merge(
                gdf_vpu[[div_col, "geometry"]].drop_duplicates(),
                on=div_col,
                how="right",
            )

            # convert df_selected to a real GeoDataFrame for plotting
            df_selected = gpd.GeoDataFrame(
                df_selected,
                geometry="geometry",
                crs=gdf_vpu.crs,
            )

        # plot the selected formulations
        columns_to_plot = cc.plots.get("columns_to_plot", None)
        if columns_to_plot is None:
            columns_to_plot = ["formulation", score_method, "summary_score", "cost"]
        plot_dict = {
            "vpu": vpu,
            "var_str": "Formulation Selection",
            "columns": columns_to_plot,
        }
        cc.plot_data(df_selected, plot_dict)


def select_formulation(
    config: cs.Config,
    vpu: str,
    df_score: pd.DataFrame,
    gdf_vpu: Optional[gpd.GeoDataFrame],
) -> pd.DataFrame:
    """Head function to select formulations for catchments in the VPU.

    Args:
        config : cs.Config
            Configuration object containing settings for the regionalization.
        vpu : str
            Vector Processing Unit (VPU) for which to select formulations.
        df_score : pd.DataFrame
            DataFrame containing summary scores for each formulation and calibrated basin.
        gdf_vpu : Optional[gpd.GeoDataFrame]
            GeoDataFrame of the VPU polygons, used for spatial operations where needed.

    Returns:
        pd.DataFrame
            DataFrame with selected formulations and their scores.

    """
    # get formulation costs from the configuration
    formulation_costs = _get_formulation_costs(config.formulation_cost)
    cost_dict = formulation_costs if config.general.consider_cost else None

    if config.general.calib_basins_only:
        # select formulations for donor basins only
        df_formulation = select_formulation_donors_only(
            config, vpu, df_score, cost_dict
        )
    else:
        # select formulations for all catchments in the VPU
        df_formulation = select_formulation_all(
            config, vpu, df_score, gdf_vpu, cost_dict
        )

    # add vpu column to the formulation DataFrame
    df_formulation["vpu"] = vpu

    # rearrange the columns in the formulation DataFrame
    div_col = getattr(config.general.id_col, "divide", "div_id")
    score_method = config.spatial_unit.best_formulation.method.lower()
    columns = [
        "vpu",
        div_col,
        "formulation",
        "huc_id",
        score_method,
        "summary_score",
        "cost",
        "num_gages",
    ]
    method1 = config.spatial_unit.basin_fill_method.lower()
    output_columns = (
        columns + ["upscale_huc"] if method1 == "upscaling" else columns + ["distances"]
    )
    df_formulation = df_formulation.reindex(
        columns=[c for c in output_columns if c in df_formulation.columns]
    )
    df_formulation[div_col] = df_formulation[div_col].astype("string")

    # check that each gage has a unique formulation assigned
    df_form_gage, df_formulation = check_gage_formulation_uniqueness(
        df_formulation, config
    )

    # save the selected formulations to the output file
    save_formulation_results(df_formulation, df_form_gage, config, vpu)

    # plot the selected formulations
    plot_formulation_results(df_formulation, config, vpu, gdf_vpu)

    return df_formulation
