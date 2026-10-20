"""Named presets bundling the column and layer naming conventions for known hydrofabric datasets.

hydrofabric_profiles.py

A profile only seeds *defaults* for the `id_col` (FieldCrosswalk) and `layer_name`
(LayerCrosswalk) sections of the general configuration. Any field a user sets explicitly
in their config always takes precedence over the profile's value for that same field, so
selecting a profile is never an all-or-nothing choice.

Only one profile is defined here: the conventions already used as this package's built-in
defaults (Lynker Spatial NextGen Hydrofabric v2.2, as used in NWM regionalization runs).
Teams adopting a different hydrofabric dataset or version should add a new entry below
once they have confirmed the actual column/layer names for that dataset, or simply set
`id_col` / `layer_name` directly in their config without naming a profile at all.

Functions:
    - get_profile: Look up a hydrofabric profile by name, raising a clear error if unknown.

"""

from typing import Any, Dict

HYDROFABRIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "nwm_ngen_hfv2.2": {
        "description": (
            "Lynker Spatial NextGen Hydrofabric v2.2 "
            "(https://lynker-spatial.s3-us-west-2.amazonaws.com/hydrofabric/v2.2/hfv2.2-data_model.html), "
            "as used in NWM regionalization runs. Matches this package's built-in defaults."
        ),
        "id_col": {
            "divide": "div_id",
            "gage": "gage_id",
            "huc12": "huc_12",
            "vpu": "vpu_id",
            "drainage_area": "area_sqkm",
        },
        "layer_name": {
            "huc12": "WBDSnapshot_National",
            "ngen": "divides",
        },
    },
}


def get_profile(name: str) -> Dict[str, Any]:
    """Look up a hydrofabric profile by name.

    Args:
        name: Name of the hydrofabric profile, as defined in HYDROFABRIC_PROFILES.

    Returns:
        Dictionary with (a subset of) 'id_col' and 'layer_name' keys, mapping to the
        default values for that profile.

    Raises:
        ValueError: If no profile with the given name is registered.

    """
    try:
        return HYDROFABRIC_PROFILES[name]
    except KeyError:
        valid = sorted(HYDROFABRIC_PROFILES)
        raise ValueError(
            f"Unknown hydrofabric_profile '{name}'. Available profiles: {valid}. "
            "Add a new entry to HYDROFABRIC_PROFILES for your hydrofabric dataset, or omit "
            "'hydrofabric_profile' and set 'id_col'/'layer_name' directly in the config."
        ) from None
