"""Parser for the ``<...Configurations>`` / ``<objectChange>`` blocks of a
Farming Simulator vehicle/placeable config XML.

It extracts the store-configuration structure as a plain data model for the
in-Blender config preview. Only the visually relevant ``objectChange``
attributes are kept (visibility, translation, rotation, scale, shaderParameter);
physics/collision attributes (mass, compoundChild, rigidBodyType, centerOfMass)
have no visual effect in Blender and are ignored.

The ``node`` references inside ``objectChange`` are i3dMapping ids; this module
keeps them as raw ids - resolving them to Blender objects is the apply engine's
job (see the importer). Attribute semantics follow the game's own parser
(dataS/scripts/utils/ObjectChangeUtil.lua).
"""

from dataclasses import dataclass, field
from typing import List
import re
import xml.etree.ElementTree as ET


# Configuration types registered in the game's ConfigurationManager
# (g_(vehicle|placeable)ConfigurationManager:addConfigurationType). ONLY these are
# real store configurations. Any other "<xxxConfigurations>" element is a
# specialization's own data block (connectionHose, powerTakeOff, differential,
# consumer, ...) and must not be shown as a store config. See store_config_spec.md
# (extracted from all addConfigurationType calls in the game source).
_REGISTERED_VEHICLE = {
    "baseColor", "designColor", "rimColor", "wrappingColor", "motor", "wheel",
    "treeSaplingType", "vehicleType", "design", "attacherJoint", "inputAttacherJoint",
    "frontloader", "cover", "folding", "cylindered", "fillUnit", "fillVolume",
    "component", "animation", "beaconLight", "ai", "workArea", "workMode",
    "powerConsumer", "dischargeable", "pipe", "plow", "roller", "baler", "trailer",
    "winch", "logGrab", "ridgeMarker", "tensionBelts", "slopeCompensation",
    "variableWorkWidth", "vineCutter", "woodHarvesterLight", "groundAdjustedNode",
    "enterablePassenger", "multipleItemPurchaseAmount", "automaticArmControlForwarder",
    "automaticArmControlHarvester", "wrappingAnimation", "cropSensor", "manureSensor",
    "pulseWidthModulation", "weedSpotSpray",
}
_REGISTERED_PLACEABLE = {"color", "incomePerHour", "solarPanels"}
# design%d / designColor%d are registered dynamically (design2..design7, ...).
_DYNAMIC_RE = re.compile(r"(?:design|designColor)\d+$")


def _is_registered_config(config_name, root_tag):
    """True if *config_name* is a registered store-configuration type for this object
    kind (vehicle vs placeable), including the dynamic design%d / designColor%d
    variants."""
    if _DYNAMIC_RE.match(config_name):
        return True
    if root_tag == "placeable":
        return config_name in _REGISTERED_PLACEABLE
    return config_name in _REGISTERED_VEHICLE


# objectChange attributes with a visual effect in Blender. Kept as raw strings;
# the apply engine parses booleans ("true"/"false") and vectors ("x y z").
_VIS_ATTRS = (
    "visibilityActive", "visibilityInactive",
    "translationActive", "translationInactive",
    "rotationActive", "rotationInactive",
    "scaleActive", "scaleInactive",
    "shaderParameter", "sharedShaderParameter",
    "shaderParameterActive", "shaderParameterInactive",
)


@dataclass
class ObjectChange:
    """One ``<objectChange>`` entry, reduced to its visual attributes."""
    node: str                                    # i3dMapping id
    attrs: dict = field(default_factory=dict)    # subset of _VIS_ATTRS present


@dataclass
class MaterialChange:
    """One ``<material>`` swap inside a configuration option."""
    slot: str                                    # materialSlotName
    template: str                                # materialTemplateName
    color_only: bool = False                     # materialTemplateUseColorOnly


@dataclass
class ConfigOption:
    """One option of a configuration type (e.g. one ``<designConfiguration>``)."""
    label: str
    changes: List[ObjectChange] = field(default_factory=list)
    materials: List["MaterialChange"] = field(default_factory=list)
    # isSelectable="false": the option exists in-game but is not user-choosable
    # (it is auto-applied by a dependency, e.g. the New-Holland/Steyr white rim
    # that follows the design brand). Kept in the list so option indices and the
    # inactive-visibility pass stay correct, but not offered as a menu button.
    selectable: bool = True
    # isDefault="true": the option pre-selected in the shop config screen.
    is_default: bool = False
    # params="60" / "2|4.5": values the game substitutes into the l10n name's
    # %s placeholders. Several options often share ONE l10n key and differ
    # only here (farmall120C: four $l10n_configuration_frontWeightX entries
    # with params 60/220/300/380) - without the params their UI labels are
    # indistinguishable (#12).
    params: str = ""


@dataclass
class ConfigType:
    """A configuration type that carries visual object changes."""
    tag: str                                     # e.g. "designConfigurations"
    name: str                                    # e.g. "design"
    options: List[ConfigOption] = field(default_factory=list)
    # Index of the option the game pre-selects (getDefaultConfigIdFromItems):
    # first isDefault+selectable, else first selectable, else 0. NOT always 0.
    default_index: int = 0


def _default_index(options):
    """The game's default option (ConfigurationUtil.getDefaultConfigIdFromItems),
    0-based: first isDefault that is selectable, else first selectable, else 0."""
    for i, o in enumerate(options):
        if o.is_default and o.selectable:
            return i
    for i, o in enumerate(options):
        if o.selectable:
            return i
    return 0


def parse_configurations(filepath) -> List[ConfigType]:
    """Return the configuration types that carry visual ``objectChange`` entries.

    Options are returned in document order (index 0 = first option, which is the
    in-game default). Types and options without any visual object change are
    omitted. Returns an empty list on a parse error or when the file has no such
    configurations.
    """
    try:
        root = ET.parse(str(filepath)).getroot()
    except (ET.ParseError, OSError):
        return []

    types: List[ConfigType] = []
    root_tag = root.tag
    for cfgs in root.iter():
        if not cfgs.tag.endswith("Configurations"):
            continue
        config_name = cfgs.tag[:-len("Configurations")]
        # Only registered store-configuration types (skips connectionHose,
        # powerTakeOff, differential, consumer, ... which are not shop configs).
        if not _is_registered_config(config_name, root_tag):
            continue
        options: List[ConfigOption] = []
        has_visual = False
        for opt in list(cfgs):
            if not opt.tag.endswith("Configuration"):
                continue
            changes: List[ObjectChange] = []
            for oc in opt.iter("objectChange"):
                node = oc.get("node")
                if not node:
                    continue
                attrs = {a: oc.get(a) for a in _VIS_ATTRS if oc.get(a) is not None}
                if attrs:
                    changes.append(ObjectChange(node=node, attrs=attrs))
            materials = []
            for mc in opt:
                if mc.tag != "material" or not mc.get("materialTemplateName"):
                    continue
                slot = mc.get("materialSlotName")
                if not slot:
                    continue
                materials.append(MaterialChange(
                    slot=slot,
                    template=mc.get("materialTemplateName"),
                    color_only=(mc.get("materialTemplateUseColorOnly") == "true"),
                ))
            label = (opt.get("name") or opt.get("vehicleName")
                     or opt.get("title") or opt.get("saveId") or "Option")
            options.append(ConfigOption(
                label=label, changes=changes, materials=materials,
                selectable=(opt.get("isSelectable") != "false"),
                is_default=(opt.get("isDefault") == "true"),
                params=(opt.get("params") or "")))
            if changes or materials:
                has_visual = True
        # Keep ALL options of a type that has at least one visual option, so
        # No/Yes (and every alternative) remain switchable. The type label uses
        # the store title when present (e.g. "Design Line", "Black Beauty").
        if has_visual:
            name = cfgs.get("title") or config_name
            types.append(ConfigType(tag=cfgs.tag, name=name, options=options,
                                     default_index=_default_index(options)))
    return types


def to_dict(types):
    """Serialise the parsed config types to plain JSON-able lists/dicts."""
    return [
        {
            "tag": t.tag,
            "name": t.name,
            "default": t.default_index,
            "options": [
                {
                    "label": o.label,
                    "changes": [{"node": c.node, "attrs": c.attrs} for c in o.changes],
                    "materials": [{"slot": m.slot, "template": m.template,
                                   "color_only": m.color_only} for m in o.materials],
                    "selectable": o.selectable,
                    "is_default": o.is_default,
                    "params": o.params,
                }
                for o in t.options
            ],
        }
        for t in types
    ]


def _set_label(name, params):
    """Display label for one <configurationSet>: fill %s placeholders in *name*
    from the pipe-separated *params*, and shorten unresolved $l10n_ tokens (we
    cannot resolve the game's localisation here)."""
    label = name or "Set"
    if params:
        for p in params.split("|"):
            label = label.replace("%s", p, 1)
    label = re.sub(r"\$l10n_(?:unit_|configuration_value|configuration_)?", "",
                   label)
    # FS unit l10n keys end in "Short" (mShort, literShort, ...) -> drop it.
    label = re.sub(r"\b([a-zA-Z]{1,6})Short\b", r"\1", label)
    return label.strip() or "Set"


def parse_configuration_sets(filepath):
    """Parse the vehicle's single top-level <configurationSets> preset chooser.

    Returns ``{"title": str|None, "controlled": [config_name, ...],
    "sets": [{"label": str, "is_default": bool, "configs": {config_name: idx0}}]}``
    with indices converted to 0-based, or None when absent. A set pins several
    sub-configurations at once (e.g. a working-width preset). All 93 base-game
    vehicles that use this have exactly one block (verified)."""
    try:
        root = ET.parse(str(filepath)).getroot()
    except (ET.ParseError, OSError):
        return None
    csel = root.find("configurationSets")
    if csel is None:
        return None
    sets = []
    controlled = set()
    for s in csel.findall("configurationSet"):
        configs = {}
        for c in s.findall("configuration"):
            nm, idx = c.get("name"), c.get("index")
            if not nm or idx is None:
                continue
            try:
                configs[nm] = int(idx) - 1           # 1-based -> 0-based
            except ValueError:
                continue
            controlled.add(nm)
        if configs:
            sets.append({"label": _set_label(s.get("name"), s.get("params")),
                         "is_default": (s.get("isDefault") == "true"),
                         "configs": configs})
    if not sets:
        return None
    return {"title": csel.get("title"), "controlled": sorted(controlled),
            "sets": sets}
