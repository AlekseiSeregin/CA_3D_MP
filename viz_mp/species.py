"""Helpers for role-based species access in viz_mp."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def get_field(item: Any, key: str, default: Any = None) -> Any:
    if item is None:
        return default
    upper = str(key).upper()
    if isinstance(item, dict):
        if key in item:
            return item[key]
        if upper in item:
            return item[upper]
        return default
    v = getattr(item, upper, None)
    if v is not None:
        return v
    return getattr(item, key, default)


def get_element(item: Any) -> Optional[str]:
    e = get_field(item, "element", None)
    if not e:
        return None
    return str(e)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return []


def inward_species(cfg: Any) -> List[Any]:
    return _as_list(getattr(cfg, "OXIDANTS", None))


def outward_species(cfg: Any) -> List[Any]:
    return _as_list(getattr(cfg, "ACTIVES", None))


def product_species(cfg: Any) -> List[Any]:
    return _as_list(getattr(cfg, "PRODUCTS", None))


def role_elements(cfg: Any) -> Dict[str, List[str]]:
    return {
        "inward": [e for e in (get_element(s) for s in inward_species(cfg)) if e],
        "outward": [e for e in (get_element(s) for s in outward_species(cfg)) if e],
        "product": [e for e in (get_element(s) for s in product_species(cfg)) if e],
    }
