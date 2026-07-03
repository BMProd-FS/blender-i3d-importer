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

Colour configurations (VehicleConfigurationItemColor /
PlaceableConfigurationItemColor) are parsed even when no option carries an
objectChange or material entry: their options define a colour / material
template (``#color`` / ``#materialTemplateName``) that the preview applies to
the target slots declared on the TYPE level (direct ``<material>`` children -
vehicles use ``materialSlotName`` (enyaq: skodaEnyaq_baseColor_mat),
placeables ``slotName`` (rudolfHormann: mainColor_mat)).
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

# Colour-type store configurations (handler VehicleConfigurationItemColor /
# PlaceableConfigurationItemColor). Their options are colours, not object
# changes, so they are parsed even when no option has a visual change (the
# enyaq baseColor section has neither - it was invisible before this).
_COLOR_VEHICLE = {"baseColor", "designColor", "rimColor", "wrappingColor"}
_COLOR_DYNAMIC_RE = re.compile(r"designColor\d+$")


# The game's generated default-colour palettes (useDefaultColors="true"):
# brandMaterialTemplates names, dumped 2026-07-03 from the RUNNING game
# (VehicleConfigurationItemColor.DEFAULT_COLORS via a one-shot log mod) -
# the Lua table constants are stripped from every decompile/luadoc dump.
# DEFAULT_COLORS_PATCH_1_2 (savegame migration) is identical and omitted.
DEFAULT_COLORS_VEHICLE = (
    "SHARED_WHITE2", "SHARED_SILVER", "SHARED_GREYLIGHT", "SHARED_GREY",
    "SHARED_GREYDARK", "SHARED_BLACKONYX", "SHARED_BLACKJET",
    "JOHNDEERE_YELLOW1", "JCB_YELLOW1", "CHALLENGER_YELLOW1",
    "SCHOUTEN_ORANGE1", "FENDT_RED1", "CASEIH_RED1", "MASSEYFERGUSON_RED",
    "HARDI_RED", "NEWHOLLAND_BLUE2", "RABE_BLUE1", "LEMKEN_BLUE1",
    "NEWHOLLAND_BLUE1", "BOECKMANN_BLUE1", "GOLDHOFER_BLUE",
    "SHARED_BLUENAVY", "LIZARD_PURPLE1", "VALTRA_GREEN2", "DEUTZ_GREEN5",
    "JOHNDEERE_GREEN1", "FENDT_NEWGREEN1", "FENDT_OLDGREEN1", "KOTTE_GREEN2",
    "CLAAS_GREEN1", "LIZARD_OLIVE1", "LIZARD_ECRU1", "SHARED_BROWN",
    "SHARED_REDCRIMSON", "LIZARD_PINK1",
)
# Placeable palette = vehicle palette + SHARED_BEIGE at position 2 (dumped
# from PlaceableConfigurationItemColor.DEFAULT_COLORS, same session).
DEFAULT_COLORS_PLACEABLE = (
    DEFAULT_COLORS_VEHICLE[:1] + ("SHARED_BEIGE",) + DEFAULT_COLORS_VEHICLE[1:]
)


def _is_registered_config(config_name, root_tag):
    """True if *config_name* is a registered store-configuration type for this object
    kind (vehicle vs placeable), including the dynamic design%d / designColor%d
    variants."""
    if _DYNAMIC_RE.match(config_name):
        return True
    if root_tag == "placeable":
        return config_name in _REGISTERED_PLACEABLE
    return config_name in _REGISTERED_VEHICLE


def _is_color_config(config_name, root_tag):
    """True if this registered config type is a colour configuration."""
    if root_tag == "placeable":
        return config_name == "color"
    return (config_name in _COLOR_VEHICLE
            or bool(_COLOR_DYNAMIC_RE.match(config_name)))


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
    """One ``<material>`` entry - either a swap inside a configuration option,
    or a colour-target declaration on the type level (colour configs declare
    their target slots ONCE as direct children of the ``<...Configurations>``
    element; vehicles use ``materialSlotName``, placeables ``slotName``).

    ``template`` may be empty: the colour/template then comes from the
    selected colour option, or from the configuration referenced by one of
    the use_* flags (VehicleConfigurationDataMaterial.onLoadFinished; the
    enyaq rim option "Base Color" relies on use_base_color)."""
    slot: str                                    # materialSlotName / slotName
    template: str = ""                           # materialTemplateName
    color_only: bool = False                     # materialTemplateUseColorOnly
    use_base_color: bool = False                 # colour of the baseColor config
    use_design_color_index: int = 0              # 1-16; designColor[n] colour
    use_rim_color: bool = False                  # colour of the rimColor config


def _parse_material(mc):
    """Parse one ``<material>`` element (option- or type-level) into a
    MaterialChange, or None when it has no target slot."""
    slot = mc.get("materialSlotName") or mc.get("slotName")
    if not slot:
        return None
    try:
        udci = int(mc.get("useDesignColorIndex") or 0)
    except ValueError:
        udci = 0
    return MaterialChange(
        slot=slot,
        template=mc.get("materialTemplateName") or "",
        color_only=(mc.get("materialTemplateUseColorOnly") == "true"),
        use_base_color=(mc.get("useBaseColor") == "true"),
        use_design_color_index=udci,
        use_rim_color=(mc.get("useRimColor") == "true"),
    )


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
    # Colour-option data (VehicleConfigurationItemColor.loadFromXML): ``color``
    # is an "r g b" string OR a material-template name; ``template`` is
    # #materialTemplateName (defines the applied look; chrome/silver template
    # names also drive the game's uiColor heuristics). isMetallic/isMat only
    # pick the finish icon in the UI. All raw strings - resolution against
    # brandMaterialTemplates happens in the preview.
    color: str = ""
    ui_color: str = ""
    template: str = ""
    is_metallic: bool = False
    is_mat: bool = False
    price: str = ""


@dataclass
class ConfigType:
    """A configuration type that carries visual object changes."""
    tag: str                                     # e.g. "designConfigurations"
    name: str                                    # e.g. "design"
    options: List[ConfigOption] = field(default_factory=list)
    # Index of the option the game pre-selects (getDefaultConfigIdFromItems):
    # first isDefault+selectable, else first selectable, else 0. NOT always 0.
    default_index: int = 0
    # Colour configuration (baseColor / designColor%d / rimColor /
    # wrappingColor, placeable color): parsed even without objectChanges.
    is_color: bool = False
    # Colour target slots declared on the TYPE level; the selected option's
    # colour/template is applied to each (enyaq: skodaEnyaq_baseColor_mat).
    color_slots: List["MaterialChange"] = field(default_factory=list)
    # useDefaultColors="true": the game appends its generated DEFAULT_COLORS
    # palette (~35 brand colours on the default_color_template base) plus a
    # "Custom Color" entry. Not expanded here - the palette list lives in a
    # stripped Lua table; see the colour-picker feature notes.
    use_default_colors: bool = False
    default_color_template: str = "calibratedPaint"
    default_color_index: int = 0                 # 1-based from XML, 0 = unset
    color_price: str = ""                        # price of generated colours


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
    """Return the configuration types that carry visual ``objectChange`` entries,
    plus every colour configuration type (their options are colours - visual by
    definition, even without objectChanges).

    Options are returned in document order (index 0 = first option, which is the
    in-game default). Non-colour types and options without any visual object
    change are omitted. Returns an empty list on a parse error or when the file
    has no such configurations.
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
        is_color = _is_color_config(config_name, root_tag)
        # Colour target slots on the type level (direct <material> children).
        color_slots: List[MaterialChange] = []
        for mc in cfgs.findall("material"):
            m = _parse_material(mc)
            if m is not None:
                color_slots.append(m)
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
            for mc in opt.findall("material"):
                m = _parse_material(mc)
                if m is None:
                    continue
                # Entries without a template are meaningful only when their
                # colour comes from elsewhere: a use_* reference, or - in a
                # colour config - the selected option itself (enyaq interior
                # slots with materialTemplateUseColorOnly and no template).
                if (m.template or m.use_base_color or m.use_design_color_index
                        or m.use_rim_color or is_color):
                    materials.append(m)
            label = (opt.get("name") or opt.get("vehicleName")
                     or opt.get("title") or opt.get("saveId")
                     or opt.get("materialTemplateName") or opt.get("color")
                     or "Option")
            options.append(ConfigOption(
                label=label, changes=changes, materials=materials,
                selectable=(opt.get("isSelectable") != "false"),
                is_default=(opt.get("isDefault") == "true"),
                params=(opt.get("params") or ""),
                color=(opt.get("color") or ""),
                ui_color=(opt.get("uiColor") or ""),
                template=(opt.get("materialTemplateName") or ""),
                is_metallic=(opt.get("isMetallic") == "true"),
                is_mat=(opt.get("isMat") == "true"),
                price=(opt.get("price") or "")))
            if changes or materials:
                has_visual = True
        # Keep ALL options of a type that has at least one visual option, so
        # No/Yes (and every alternative) remain switchable. Colour configs are
        # kept whenever they have options at all: their colours apply to the
        # type-level slots, and even slot-less colour types matter as the
        # target of use_base_color / use_design_color_index references. The
        # type label uses the store title when present (e.g. "Design Line").
        if has_visual or (is_color and options):
            name = cfgs.get("title") or config_name
            try:
                dci = int(cfgs.get("defaultColorIndex") or 0)
            except ValueError:
                dci = 0
            types.append(ConfigType(
                tag=cfgs.tag, name=name, options=options,
                default_index=_default_index(options),
                is_color=is_color, color_slots=color_slots,
                use_default_colors=(cfgs.get("useDefaultColors") == "true"),
                default_color_template=(
                    cfgs.get("defaultColorMaterialTemplateName")
                    or "calibratedPaint"),
                default_color_index=dci,
                color_price=(cfgs.get("price") or "")))
    return types


def _mat_d(m):
    """Serialise one MaterialChange."""
    return {"slot": m.slot, "template": m.template, "color_only": m.color_only,
            "use_base_color": m.use_base_color,
            "use_design_color_index": m.use_design_color_index,
            "use_rim_color": m.use_rim_color}


def to_dict(types):
    """Serialise the parsed config types to plain JSON-able lists/dicts."""
    return [
        {
            "tag": t.tag,
            "name": t.name,
            "default": t.default_index,
            "is_color": t.is_color,
            "color_slots": [_mat_d(m) for m in t.color_slots],
            "use_default_colors": t.use_default_colors,
            "default_color_template": t.default_color_template,
            "default_color_index": t.default_color_index,
            "color_price": t.color_price,
            "options": [
                {
                    "label": o.label,
                    "changes": [{"node": c.node, "attrs": c.attrs} for c in o.changes],
                    "materials": [_mat_d(m) for m in o.materials],
                    "selectable": o.selectable,
                    "is_default": o.is_default,
                    "params": o.params,
                    "color": o.color,
                    "ui_color": o.ui_color,
                    "template": o.template,
                    "is_metallic": o.is_metallic,
                    "is_mat": o.is_mat,
                    "price": o.price,
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
