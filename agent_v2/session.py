"""Session state: messages + accumulated requirements + candidate house IDs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_sessions: dict[str, "Session"] = {}


@dataclass
class Session:
    session_id: str
    initialized: bool = False
    messages: list[dict] = field(default_factory=list)
    # Accumulated search requirements across turns
    requirements: dict[str, Any] = field(default_factory=dict)
    # Current top-5 candidate house IDs shown to user
    candidates: list[str] = field(default_factory=list)
    # All house IDs from the last search (before top-5 cut)
    all_results: list[str] = field(default_factory=list)
    # Tracking for rent/terminate
    rented_house: tuple[str, str] | None = None  # (house_id, platform)
    # Turn counter
    turn: int = 0

    def append_msg(self, *msgs: dict) -> None:
        self.messages.extend(msgs)

    def merge_requirements(self, new: dict[str, Any]) -> None:
        """Merge new requirements into accumulated state."""
        for k, v in new.items():
            if v is None:
                continue
            if k == "tags_require":
                existing = self.requirements.get("tags_require", [])
                for t in v:
                    if t not in existing:
                        existing.append(t)
                self.requirements["tags_require"] = existing
            elif k == "tags_exclude":
                existing = self.requirements.get("tags_exclude", [])
                for t in v:
                    if t not in existing:
                        existing.append(t)
                self.requirements["tags_exclude"] = existing
            elif k == "field_filters":
                existing = self.requirements.get("field_filters", [])
                for f in v:
                    if f not in existing:
                        existing.append(f)
                self.requirements["field_filters"] = existing
            elif k == "max_price" and v is not None:
                old = self.requirements.get("max_price")
                self.requirements["max_price"] = v if old is None else min(old, v)
            elif k == "min_price" and v is not None:
                old = self.requirements.get("min_price")
                self.requirements["min_price"] = v if old is None else max(old, v)
            else:
                self.requirements[k] = v

    def build_search_params(self) -> dict[str, Any]:
        """Convert accumulated requirements into get_houses_by_platform params."""
        r = self.requirements
        p: dict[str, Any] = {"page_size": 50}
        mapping = {
            "district": "district",
            "bedrooms": "bedrooms",
            "max_price": "max_price",
            "min_price": "min_price",
            "rental_type": "rental_type",
            "decoration": "decoration",
            "elevator": "elevator",
            "orientation": "orientation",
            "max_subway_dist": "max_subway_dist",
            "subway_station": "subway_station",
            "subway_line": "subway_line",
            "commute_to_xierqi_max": "commute_to_xierqi_max",
            "listing_platform": "listing_platform",
            "sort_by": "sort_by",
            "sort_order": "sort_order",
            "min_area": "min_area",
            "max_area": "max_area",
            "available_from_before": "available_from_before",
            "utilities_type": "utilities_type",
            "area": "area",
        }
        for req_key, param_key in mapping.items():
            val = r.get(req_key)
            if val is not None:
                p[param_key] = val
        return p


def get_session(session_id: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id=session_id)
    return _sessions[session_id]
