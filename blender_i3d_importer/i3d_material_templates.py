"""Resolve FS25 material templates used by design / color store configurations.

A ``<material materialTemplateName="X">`` swap inside a configuration pulls the
shader values of template X - colorScale, smoothnessScale, clearCoatIntensity, ...
following ``parentTemplate`` inheritance - which the preview then writes into the
imported material's fs25_param nodes.

Templates live in two shared files (both use flat ``<template name=...>`` entries):
  - data/shared/brandMaterialTemplates.xml   (brand colors, e.g. FENDT_RED1)
  - data/shared/detailLibrary/materialTemplates.xml  (generic, e.g. metalPaintedBlack)
"""

import os
import xml.etree.ElementTree as ET

# Template attributes that correspond to fs25 material CustomParameters
# (these have a matching fs25_param node in our debug materials).
PARAM_ATTRS = ("colorScale", "smoothnessScale", "clearCoatIntensity",
               "clearCoatSmoothness", "porosity", "metalness")
# Texture attributes (phase B - texture swapping).
TEXTURE_ATTRS = ("detailDiffuse", "detailNormal", "detailSpecular")
# Meta attributes. ``category`` ("metallic/..." vs "nonMetallic/...") drives the
# metalness/chrome look of a template and is inherited via parentTemplate.
META_ATTRS = ("category",)

_TEMPLATE_FILES = (
    os.path.join("shared", "brandMaterialTemplates.xml"),
    os.path.join("shared", "detailLibrary", "materialTemplates.xml"),
)


def _data_dir(fs25_root):
    """Return the FS25 ``data`` directory for a configured game root."""
    root = fs25_root.rstrip("/\\")
    if os.path.basename(root).lower() == "data":
        return root
    return os.path.join(root, "data")


def load_templates(fs25_root):
    """Parse both template files into a dict ``name -> attrib dict``.

    Later files do not override earlier ones (names are unique across the two).
    Returns an empty dict if the files are missing.
    """
    data = _data_dir(fs25_root)
    templates = {}
    for rel in _TEMPLATE_FILES:
        path = os.path.join(data, rel)
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        for el in root.iter("template"):
            name = el.get("name")
            if name and name not in templates:
                templates[name] = dict(el.attrib)
    return templates


def resolve_template(name, templates, _seen=None):
    """Resolve a template to its effective param/texture values.

    Follows ``parentTemplate`` (child overrides parent). Returns a dict with the
    subset of PARAM_ATTRS / TEXTURE_ATTRS that are set. Empty dict for an unknown
    template.
    """
    el = templates.get(name)
    if el is None:
        return {}
    _seen = _seen if _seen is not None else set()
    if name in _seen:
        return {}
    _seen.add(name)

    parent = el.get("parentTemplate")
    out = resolve_template(parent, templates, _seen) if parent else {}
    out = dict(out)
    for a in PARAM_ATTRS + TEXTURE_ATTRS + META_ATTRS:
        v = el.get(a)
        if v is not None:
            out[a] = v
    return out
