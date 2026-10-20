"""Unit tests for hydrofabric_profile / valid_vpus config generalization.

Covers BaseGeneralConfig.apply_hydrofabric_profile and check_vpu, which allow this package
to be pointed at a hydrofabric dataset with different column/layer names and/or a different
domain/VPU partitioning scheme without a code change.
"""

import pytest
from pydantic import BaseModel

from nwm_region_mgr.formreg.config_schema import FormulationGeneralSettings
from nwm_region_mgr.utils.hydrofabric_profiles import HYDROFABRIC_PROFILES

BASE_KWARGS = dict(
    base_dir="/fake/base/dir",
    static_data_dir="/fake/static/dir",
    ngen_hydrofabric_file="vpu_01.gpkg",
)


class _GeneralWrapper(BaseModel):
    """Minimal wrapper validating `general` as a nested field, matching how pydantic
    handles Config.general in production. Passing a dict via **kwargs at the top level
    would copy it at the call boundary and never exercise pydantic's nested-model
    mode="before" validation path, where no defensive copy is guaranteed.
    """

    general: FormulationGeneralSettings


# -------------------------------hydrofabric_profile-------------------------------
@pytest.mark.unit
def test_no_profile_uses_builtin_defaults():
    """Without a profile, id_col/layer_name keep their own built-in defaults."""
    settings = FormulationGeneralSettings(**BASE_KWARGS)
    assert settings.id_col.divide == "div_id"
    assert settings.id_col.gage == "gage_id"
    assert settings.layer_name.ngen == "divides"


@pytest.mark.unit
def test_known_profile_seeds_id_col_and_layer_name():
    """Selecting a known profile seeds id_col/layer_name from it."""
    settings = FormulationGeneralSettings(
        **BASE_KWARGS, hydrofabric_profile="nwm_ngen_hfv2.2"
    )
    assert settings.id_col.divide == "div_id"
    assert settings.id_col.gage == "gage_id"
    assert settings.layer_name.ngen == "divides"
    assert settings.layer_name.huc12 == "wbdsnapshot_national"


@pytest.mark.unit
def test_profile_plus_partial_override_only_overrides_that_field():
    """A field set explicitly in the config always wins over the profile's value for it."""
    settings = FormulationGeneralSettings(
        **BASE_KWARGS,
        hydrofabric_profile="nwm_ngen_hfv2.2",
        id_col={"gage": "custom_gage_col"},
    )
    assert settings.id_col.gage == "custom_gage_col"  # overridden
    assert settings.id_col.divide == "div_id"  # still seeded by the profile


@pytest.mark.unit
def test_unknown_profile_raises_clear_error():
    """An unknown profile name raises a validation error naming the available profiles."""
    with pytest.raises(Exception, match="Unknown hydrofabric_profile 'does_not_exist'"):
        FormulationGeneralSettings(
            **BASE_KWARGS, hydrofabric_profile="does_not_exist"
        )


@pytest.mark.unit
def test_profile_resolution_does_not_leak_across_reused_dicts(monkeypatch):
    """apply_hydrofabric_profile must not mutate its input, or a stale profile's values
    could leak into a later validation that reuses the same underlying dict object (e.g.
    a config-building helper constructing several Config(...) objects from one template).
    """
    monkeypatch.setitem(
        HYDROFABRIC_PROFILES,
        "other_profile",
        {"id_col": {"divide": "other_divide_col", "gage": "other_gage_col"}},
    )

    shared_general = dict(**BASE_KWARGS, hydrofabric_profile="other_profile")

    wrapper1 = _GeneralWrapper(general=shared_general)
    assert wrapper1.general.id_col.divide == "other_divide_col"
    assert "id_col" not in shared_general  # input dict must be untouched

    shared_general["hydrofabric_profile"] = "nwm_ngen_hfv2.2"
    wrapper2 = _GeneralWrapper(general=shared_general)
    assert wrapper2.general.id_col.divide == "div_id"  # not leaked from wrapper1's profile


# -------------------------------domain / valid_vpus-------------------------------
@pytest.mark.unit
def test_builtin_domains_unaffected():
    """The four built-in domains validate exactly as before."""
    settings = FormulationGeneralSettings(**BASE_KWARGS, domain="ak", vpu="19")
    assert settings.domain == "ak"
    assert settings.vpu == "19"


@pytest.mark.unit
def test_unknown_domain_without_override_raises():
    """An unrecognized domain with no valid_vpus override raises a clear, actionable error."""
    with pytest.raises(Exception, match="Unknown domain 'gl'"):
        FormulationGeneralSettings(**BASE_KWARGS, domain="gl", vpu="22")


@pytest.mark.unit
def test_unknown_domain_with_valid_vpus_override_succeeds():
    """A custom domain/partitioning scheme works once valid_vpus supplies its VPU codes."""
    settings = FormulationGeneralSettings(
        **BASE_KWARGS,
        domain="gl",
        vpu="22",
        valid_vpus={"gl": ["22", "23"]},
    )
    assert settings.domain == "gl"
    assert settings.vpu == "22"


@pytest.mark.unit
def test_invalid_vpu_for_known_domain_still_raises():
    """An out-of-range VPU for a known domain still raises, unaffected by the override field."""
    with pytest.raises(Exception, match="Invalid VPU '99' for domain 'conus'"):
        FormulationGeneralSettings(**BASE_KWARGS, domain="conus", vpu="99")


@pytest.mark.unit
def test_valid_vpus_extends_a_known_domain_rather_than_replacing_it():
    """Adding a VPU code to a known domain must not drop that domain's other built-in codes."""
    # the newly added code validates
    settings_new = FormulationGeneralSettings(
        **BASE_KWARGS, domain="conus", vpu="99", valid_vpus={"conus": ["99"]}
    )
    assert settings_new.vpu == "99"

    # an existing, unmentioned built-in code must still validate too
    settings_existing = FormulationGeneralSettings(
        **BASE_KWARGS, domain="conus", vpu="05", valid_vpus={"conus": ["99"]}
    )
    assert settings_existing.vpu == "05"
