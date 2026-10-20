"""Generate documentation for configuration schemas and sample configs used in nwm-region-mgr.

This script generates markdown documentation for the configuration schemas used in the NWM Regionalization Manager
    (nwm-region-mgr) tool. It creates example YAML configuration files and detailed markdown tables describing each
    field in the configuration schemas, including their types, descriptions, default values, and examples.

The generated markdown file is saved to `docs/source/config_builder/index.md`.
"""

import re
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from nwm_region_mgr.formreg.config_schema import (
    BestFormulation,
    FormulationCostConfig,
    FormulationGeneralSettings,
    FormulationOutputConfig,
    FormulationSpatialUnitConfig,
    FormulationSummaryScoreConfig,
    MetricConfig,
)
from nwm_region_mgr.formreg.config_schema import Config as FormConf
from nwm_region_mgr.ngen.config_schema import Config as NgenConf
from nwm_region_mgr.ngen.config_schema import (
    NgenGeneralSettings,
    NgenOutputConfig,
)
from nwm_region_mgr.parreg.config_schema import (
    HDBSCAN,
    URF,
    AlgoGeneral,
    AlgorithmConfig,
    AttrDatasetConfig,
    AvailableAttrsConfig,
    Birch,
    DonorConfig,
    GeneralConfig,
    Gower,
    KMeans,
    KMedoids,
    MetricEvalPeriod,
    MetricThreshold,
    ParameterOutputConfig,
    SnowCoverConfig,
)
from nwm_region_mgr.parreg.config_schema import Config as ParConf
from nwm_region_mgr.utils.config_utils import (
    BaseGeneralConfig,
    BaseOutputConfig,
    FieldCrosswalk,
    LayerCrosswalk,
    LoggingConfig,
)

INDENT_LEVEL = 2
YAML_COMMENT_BUFFER = 5
NO_DESCRIPTION_STR = "No description provided"
DOCS_TO_CREATE = {
    "General Configuration": {
        "description": "`config_general.yaml`: General configurations (shared by formulation & parameter regionalizations)",
        "example_file_class": (BaseGeneralConfig, "general"),
        "schemas": {
            "general": BaseGeneralConfig,
            "id_col": FieldCrosswalk,
            "layer_name": LayerCrosswalk,
            "logging": LoggingConfig,
        },
    },
    "Formulation Regionalization": {
        "description": "`config_formreg.yaml`: specific configurations for formulation regionalization (formreg)",
        "example_file_class": (FormConf,),
        "schemas": {
            "general": FormulationGeneralSettings,
            "general.logging": LoggingConfig,
            "spatial_unit": FormulationSpatialUnitConfig,
            "spatial_unit.best_formulation": BestFormulation,
            "summary_score": FormulationSummaryScoreConfig,
            "summary_score.metric_eval_period": MetricEvalPeriod,
            "summary_score.metrics": MetricConfig,
            "formulation_cost": FormulationCostConfig,
            "output": FormulationOutputConfig,
            "Type: BaseOutputConfig": BaseOutputConfig,
        },
    },
    "Parameter Regionalization": {
        "description": "`config_parreg.yaml`: specific configurations for parameter regionalization (parreg)",
        "example_file_class": (ParConf,),
        "schemas": {
            "general": GeneralConfig,
            "general.logging": LoggingConfig,
            "donor": DonorConfig,
            "donor.metric_eval_period": MetricEvalPeriod,
            "donor.metric_threshold": MetricThreshold,
            "attr_datasets": AvailableAttrsConfig,
            "Class: AttrDatasetConfig": AttrDatasetConfig,
            "snow_cover": SnowCoverConfig,
            "algorithms": AlgorithmConfig,
            "algorithms.algo_general": AlgoGeneral,
            "algorithms.gower": Gower,
            "algorithms.kmeans": KMeans,
            "algorithms.kmedoids": KMedoids,
            "algorithms.birch": Birch,
            "algorithms.hdbscan": HDBSCAN,
            "output": ParameterOutputConfig,
            "Class: BaseOutputConfig": BaseOutputConfig,
        },
    },
    "NGEN Simulation": {
        "description": "`config_ngen.yaml`: specific configurations for NGEN simulation",
        "example_file_class": (NgenConf,),
        "schemas": {
            "general": NgenGeneralSettings,
            "general.logging": LoggingConfig,
            "output": NgenOutputConfig,
            "Type: BaseOutputConfig": BaseOutputConfig,
        },
    },
    "Evaluation": {
        "description": "{{ '`config_eval.yaml`: specific configurations for evaluation. See details at [nwm-eval-mgr configuration](https://ngwpc.github.io/nwm-eval-mgr/config.html#)'.format(github_org) }}",
        "example_file_class": (),
        "schemas": {},
    },
}


def type_to_str(tp):
    """Convert a type hint to a readable string for markdown."""
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is None:  # Simple case e.g., str
        return getattr(tp, "__name__", str(tp))
    elif origin in (list, list):
        return f"list[{type_to_str(args[0])}]" if args else "list"
    elif origin in (dict, dict):
        return (
            f"dict[{type_to_str(args[0])}, {type_to_str(args[1])}]" if args else "dict"
        )
    elif origin is Literal:
        return "str = " + " \\| ".join(map(str, args))
    elif len(args) > 1:
        return " \\| ".join(type_to_str(a) for a in args)
    else:  # Catch others
        return str(tp)


def field_to_dict(field: FieldInfo, value: Any = None, omit_none: bool = True) -> dict:
    """Convert a Pydantic field to a dict for YAML/schema generation."""
    type_ = getattr(field, "annotation", Any)
    description = getattr(field, "description", None) or NO_DESCRIPTION_STR

    # Handle default or default_factory
    if value is not None:
        default_value = value
    elif getattr(field, "default_factory", None) is not None:
        try:
            default_value = field.default_factory()
        except Exception:
            default_value = "<factory>"
    elif getattr(field, "default", ...) is not ...:
        default_value = field.default
    else:
        default_value = "required"

    # Handle examples: prefer outer field examples if present
    example = getattr(field, "examples", None)
    if example is None:
        example = default_value if default_value != "required" else ""

    sub_dict = None
    # True only when sub_dict is keyed by real instance names (e.g. dataset/metric names),
    # as opposed to a dict[str, BaseModel] field's generic fallback below, which is instead
    # keyed by the model class's own field names (e.g. 'attr_list', 'attr_data_file') and
    # would render misleadingly if treated as if each key were a named instance.
    sub_dict_keyed_by_name = False

    # Handle nested BaseModel
    if isinstance(default_value, BaseModel):
        sub_dict = pydantic_to_dict(
            type(default_value),
            type(default_value),
            instance=default_value,
            outer_examples=example,
        )
    elif isinstance(type_, type) and issubclass(type_, BaseModel):
        sub_dict = pydantic_to_dict(type_, type_, outer_examples=example)
    else:
        # dict[str, BaseModel]
        origin = get_origin(type_)
        args = get_args(type_)
        if (
            origin in (dict, dict)
            and len(args) == 2
            and isinstance(args[1], type)
            and issubclass(args[1], BaseModel)
        ):
            value_type = args[1]
            # Prefer the default's keys (e.g., a non-empty default_factory); for an open
            # registry field (default {}) with no fixed keys, fall back to the example's
            # keys so illustrative entries are still documented.
            if isinstance(default_value, dict) and default_value:
                keys_source = default_value
            elif isinstance(example, dict) and example:
                keys_source = example
            else:
                keys_source = None

            if keys_source is not None:
                sub_dict_keyed_by_name = True
                # Each key's value must itself be a field-info dict (matching this
                # function's own return shape), not the raw pydantic_to_dict(...) output --
                # pydantic_dict_to_lines expects every dict_rep value it recurses into to
                # have its own 'description'/'type'/'sub_dict' keys, one level further down.
                sub_dict = {
                    k: {
                        "type": value_type,
                        "description": NO_DESCRIPTION_STR,
                        "default": None,
                        "example": example.get(k) if isinstance(example, dict) else None,
                        "sub_dict": pydantic_to_dict(
                            value_type,
                            value_type,
                            outer_examples=(
                                example.get(k) if isinstance(example, dict) else None
                            ),
                        ),
                        "sub_dict_keyed_by_name": False,
                    }
                    for k in keys_source.keys()
                }
            else:
                sub_dict = pydantic_to_dict(value_type, value_type)
    return {
        "type": type_,
        "description": description,
        "default": default_value,
        "example": example,
        "sub_dict": sub_dict,
        "sub_dict_keyed_by_name": sub_dict_keyed_by_name,
    }


def pydantic_to_dict(
    model_cls: type[BaseModel],
    base_cls: type[BaseModel],
    instance: BaseModel | None = None,
    outer_examples: Any = None,
    start_fields: list[str] = ["general"],
    end_fields: list[str] = ["output", "algorithms"],
) -> dict[str, dict]:
    """Convert a Pydantic model to dict, optionally using an instance.

    Only child-only fields are included in the 'general' section.
    Always starts with start_fields and ends with end_fields.

    """
    dict_rep = {}

    # Fields inherited from base_cls (to exclude in general)
    base_fields = set(base_cls.model_fields.keys()) if model_cls != base_cls else set()

    # Ordered field names
    all_fields = list(model_cls.model_fields.keys())
    middle_fields = [f for f in all_fields if f not in start_fields + end_fields]
    ordered_fields = start_fields + middle_fields + end_fields

    for name in ordered_fields:
        if name not in model_cls.model_fields:
            continue
        field = model_cls.model_fields[name]

        # Handle 'general' section: only child-only fields (except "logging")
        if name == "general" and model_cls != base_cls:
            general_sub_dict = {}
            for sub_name, sub_field in field.annotation.model_fields.items():
                if sub_name not in base_fields or sub_name == "logging":
                    # Pass instance value if present
                    value = (
                        getattr(instance.general, sub_name, None) if instance else None
                    )
                    general_sub_dict[sub_name] = field_to_dict(sub_field, value=value)

            dict_rep[name] = {
                "description": getattr(field, "description", NO_DESCRIPTION_STR),
                "sub_dict": general_sub_dict,
            }
            continue

        # Other sections: all fields
        value = getattr(instance, name, None) if instance else None
        dict_rep[name] = field_to_dict(field, value=value)

        # Override example with outer_examples if provided
        if isinstance(outer_examples, dict) and name in outer_examples:
            dict_rep[name]["example"] = outer_examples[name]

    return dict_rep


def dict_to_yaml(dict_rep: dict, indent: int = 0) -> list[str]:
    """Convert a python dictionary to YAML with added indent customization."""
    lines = []
    tmp_ind = " " * indent
    for i in dict_rep:
        if isinstance(dict_rep[i], dict):
            lines.append(f"{tmp_ind}{i}:")
            lines.extend(dict_to_yaml(dict_rep[i], indent + INDENT_LEVEL))
        else:
            lines.append(f"{tmp_ind}{i}: {str(dict_rep[i])}")
    return lines


def is_dict_of_basemodel(field_dict: dict) -> bool:
    """Check if a field dict represents a dict[str, BaseModel]-like field."""
    type_ = field_dict.get("type")
    if type_ is None:
        return False
    origin = get_origin(type_)
    args = get_args(type_)
    return (
        origin in (dict, dict)
        and len(args) == 2
        and isinstance(args[1], type)
        and issubclass(args[1], BaseModel)
    )


def pydantic_dict_to_lines(dict_rep: dict, indent: int = 0) -> list[str]:
    """Convert a pydantic model to lines in the YAML format."""
    lines = []
    tmp_ind = " " * indent
    com_buffer = " " * YAML_COMMENT_BUFFER

    for k, v in dict_rep.items():
        comment = (
            f"{com_buffer}# {v.get('description')}"
            if v.get("description") != NO_DESCRIPTION_STR
            else ""
        )

        # If sub_dict exists, recurse -- except for a dict[str, BaseModel] field whose
        # sub_dict is NOT keyed by real instance names (i.e., it's the generic fallback,
        # keyed by the model class's own field names instead), since recursing into that
        # would misleadingly render those field names as if each were a named instance.
        if v.get("sub_dict") is not None and (
            not is_dict_of_basemodel(v) or v.get("sub_dict_keyed_by_name")
        ):
            lines.append(f"{tmp_ind}{k}:{comment}")
            lines.extend(pydantic_dict_to_lines(v["sub_dict"], indent + INDENT_LEVEL))
            continue

        # Skip fields without examples
        if v.get("example") is None:
            continue

        # Convert value to string
        val = v["example"]
        if isinstance(val, str):
            val = f"'{val}'"
        elif isinstance(val, dict):
            lines.append(f"{tmp_ind}{k}:{comment}")
            lines.extend(dict_to_yaml(val, indent + INDENT_LEVEL))
            continue

        lines.append(f"{tmp_ind}{k}: {val}{comment}")

    return lines


def justify_yaml_comments(lines: list[str]):
    """Update a YAML file so that comments all start at same column."""
    max_len = 0
    for i in lines:
        cur_len = len(i.split(" #")[0])
        max_len = max([cur_len, max_len])
    for ind, i in enumerate(lines):
        cur_len = len(i.split(" #")[0])
        lines[ind] = i.replace(
            " #", " #" + ("-" * ((max_len + YAML_COMMENT_BUFFER) - cur_len))
        )

    return lines


def generate_yaml_template(model_cls: type[BaseModel], top_key: str = None) -> str:
    """Generate YAML file using examples and descriptions."""
    # Convert BaseModel to dict for easier manipulation
    dict_rep = pydantic_to_dict(
        model_cls,
        BaseGeneralConfig,
    )

    # Build initial yaml lines
    if top_key is not None:
        lines = [f"{top_key}:"]
        indent = INDENT_LEVEL
    else:
        lines = []
        indent = 0
    lines.extend(pydantic_dict_to_lines(dict_rep, indent))

    # comment out comments justification for now, as it can create very long lines and
    # hence make comments in the sample yaml files not readily viewable
    # lines = justify_yaml_comments(lines)
    return "\n".join(lines)


def generate_markdown_table(model_cls: type[BaseModel]) -> str:
    """Convert a pydantic model to a markdown table."""
    # Convert BaseModel to dict for easier manipulation
    dict_rep = pydantic_to_dict(model_cls, BaseGeneralConfig)

    # remove those fields that are not child-only in 'general' section
    if model_cls in (FormulationGeneralSettings, GeneralConfig, NgenGeneralSettings):
        base_fields = set(BaseGeneralConfig.model_fields.keys())
        for k in list(dict_rep.keys()):
            if k in base_fields:
                dict_rep.pop(k)

    # Make markdown table
    headers = ["Field", "Type(s)", "Description", "Default", "Example(s)"]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for name, field_dict in dict_rep.items():
        table.append(
            f"| {name} | {type_to_str(field_dict['type'])} | "
            f"{field_dict['description']} | {field_dict['default']} | {field_dict['example']} |"
        )

    return "\n".join(table)


def myst_anchor(title: str) -> str:
    """Convert a heading to MyST-style anchor ID."""
    title = title.strip().lower()

    # Replace any character that is NOT a-z, 0-9 with dash
    title = re.sub(r"[^a-z0-9]+", "-", title)

    # Collapse multiple consecutive dashes
    title = re.sub(r"-+", "-", title)

    # Strip leading/trailing dashes
    title = title.strip("-")

    return title


def generate_toc_from_markdown(md_text: str) -> str:
    """Scan generated markdown, find headings, and return a TOC block."""
    toc_lines = ["## Table of Contents", ""]
    heading_stack = []

    for line in md_text.splitlines():
        m = re.match(r"^(#{2,6})\s+(.*)", line)
        if not m:
            continue

        level = len(m.group(1))
        title = m.group(2).strip()

        # Remove headings deeper than the current heading level.
        heading_stack = heading_stack[: level - 2]

        # Build a hierarchical anchor from the current heading path.
        heading_stack.append(title)
        anchor = myst_anchor("-".join(heading_stack))

        indent = "  " * (level - 2)
        toc_lines.append(f"{indent}- [{title}](#{anchor})")

    return "\n".join(toc_lines) + "\n\n"


def main(docs_to_create: dict) -> None:
    """Create markdown file documentation for the specified pydantic models."""
    # Static intro paragraph
    intro_block = ["# Regionalization Configuration\n"]
    intro_block.append("## Introduction\n")
    intro_block.append(
        "This section provides detailed documentation for the configuration files used "
        "in the NWM Regionalization Manager (nwm-region-mgr) tool. The configuration "
        "files define the parameters and settings for both formulation and parameter "
        "regionalization processes.\n"
    )
    intro_block.append(
        "Example files and schemas for all configuration fields and subfields are included below. "
        "You can navigate to each config file or schema section using the tabs on the right or the "
        "table of contents below.\n"
    )
    intro_block.append(":::{note}\n")
    intro_block.append(
        "The tabs on the left will take you to the builder for each of the specific config files. "
        "Currently, only the general configuration builder is available. The builders "
        "for formulation and parameter regionalizations are still under development. In the builder, "
        "you will be prompted to enter setup information for your regionalization run, or you "
        "can scroll to the bottom to fill in default values. Once done, hit 'download' to save "
        "the generated configuration YAML file to your local system.\n"
    )
    intro_block.append(":::\n")

    intro_block = "\n".join(intro_block)

    # Generate sections for each config file
    lines = []

    for i, value in docs_to_create.items():
        # Generate a unique anchor for the config file section.
        doc_anchor = (
            i.replace("_", "-")
            .replace(".", "-")
            .replace(": ", "-")
            .replace(" ", "-")
            .lower()
        )

        lines.append(f"({doc_anchor})=\n")
        lines.append(f"## {i}\n")
        lines.append(f"{value.get('description', '')}\n")

        if i == "Evaluation":
            continue

        # Example file section
        example_anchor = f"{doc_anchor}-example-file"
        lines.append(f"({example_anchor})=\n")
        lines.append("### Example File\n")

        lines.append("```yaml")
        lines.append(generate_yaml_template(*value["example_file_class"]))
        lines.append("```")

        # Schemas section
        schemas_anchor = f"{doc_anchor}-schemas"
        lines.append(f"({schemas_anchor})=\n")
        lines.append("### Schemas\n")

        for j, model_cls in value["schemas"].items():
            # Generate a unique anchor using the parent config section.
            schema_anchor = (
                f"{schemas_anchor}-"
                f"{j.replace('_', '-').replace('.', '-').replace(': ', '-').lower()}"
            )

            lines.append("")
            lines.append(f"({schema_anchor})=\n")
            lines.append(f"#### {j}\n")

            # Append class information
            class_info = f"Class `{model_cls.__name__}`."

            parent_cls = [
                c
                for c in model_cls.__bases__
                if issubclass(c, BaseModel) and c not in (BaseModel, object)
            ]

            if parent_cls:
                class_info += (
                    f" Inherits `{', '.join(c.__name__ for c in parent_cls)}`."
                )

            lines.append(class_info)
            lines.append("")
            lines.append(generate_markdown_table(value["schemas"][j]))

    # Append toctree block (for left menu) at the end
    lines.append("")
    lines.append(":::{toctree}")
    lines.append(":maxdepth: 2")
    lines.append(":hidden:")
    lines.append("")
    lines.append("General<general>")
    lines.append("Formulation Regionalization<formreg>")
    lines.append("Parameter Regionalization<parreg>")
    lines.append(":::")

    # Convert to final markdown text
    md_text = "\n".join(lines)

    # Generate TOC
    toc_block = generate_toc_from_markdown(md_text)

    # Prepend intro and TOC
    md_text = intro_block + toc_block + md_text

    # Write to file
    file_path = Path(__file__).parent.parent / "source" / "config_builder" / "index.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(md_text, encoding="utf-8")
    print(f"Config schemas generated and saved to {file_path}")


if __name__ == "__main__":
    main(DOCS_TO_CREATE)
