"""
Pathfinder Profile

Loads profile.yaml into a structured, queryable object.

Why this module exists
----------------------
The first version of Pathfinder kept the candidate's facts in three written
documents and pulled four 500-character chunks out of a vector store on every
turn. That is a reasonable pattern when there is a lot of material, but for a
single resume and a one-page preference sheet it actively loses information: the model would
often see the location preferences but not the salary range, or the skills but
not the dealbreakers, depending on how the question was phrased.

Structured profile data is deterministic. Every scoring decision reads from the
same complete object every time, and the whole profile is small enough to sit
in the system prompt without retrieval. Retrieval is still used, but only for
the strategy guide, which is the one document where semantic lookup helps.

This module is also the seam for multi-user support. Nothing about Colin is
hardcoded here; swap the YAML file and the entire app retargets.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import yaml

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile.yaml")


# =============================================================================
# Small value objects
# =============================================================================

@dataclass(frozen=True)
class Skill:
    name: str
    tier: str                      # core | working | exposure
    aliases: tuple[str, ...]

    @property
    def value(self) -> float:
        """How much matching this skill is worth, relative to other skills."""
        return {"core": 1.0, "working": 0.6, "exposure": 0.3}.get(self.tier, 0.5)


@dataclass(frozen=True)
class City:
    name: str
    state: str
    col_index: float               # 100 = US average cost of living
    score: float                   # 0-1 desirability


@dataclass(frozen=True)
class RoleFamily:
    key: str
    label: str
    weight: float
    queries: tuple[str, ...]
    titles: tuple[str, ...]


@dataclass(frozen=True)
class Signal:
    phrase: str
    weight: int


# =============================================================================
# The profile
# =============================================================================

@dataclass
class Profile:
    raw: dict[str, Any]

    identity: dict[str, Any] = field(default_factory=dict)
    skills: list[Skill] = field(default_factory=list)
    role_families: dict[str, RoleFamily] = field(default_factory=dict)
    cities: dict[str, City] = field(default_factory=dict)
    green_signals: list[Signal] = field(default_factory=list)
    red_signals: list[Signal] = field(default_factory=list)

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: str = PROFILE_PATH) -> "Profile":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Profile":
        """Build a Profile from a plain dict.

        Kept separate from load() so a future web form can hand us a dict
        without ever touching the filesystem.
        """
        p = cls(raw=raw)
        p.identity = raw.get("identity", {})

        p.skills = [
            Skill(
                name=s["name"],
                tier=s.get("tier", "working"),
                aliases=tuple(a.lower() for a in s.get("aliases", [])) or (s["name"].lower(),),
            )
            for s in raw.get("skills", [])
        ]

        p.role_families = {
            key: RoleFamily(
                key=key,
                label=rf.get("label", key.replace("_", " ").title()),
                weight=float(rf.get("weight", 0.5)),
                queries=tuple(rf.get("queries", [])),
                titles=tuple(t.lower() for t in rf.get("titles", [])),
            )
            for key, rf in raw.get("role_families", {}).items()
        }

        loc = raw.get("location", {})
        for tier in loc.get("tiers", []):
            for c in tier.get("cities", []):
                city = City(
                    name=c["name"],
                    state=c.get("state", ""),
                    col_index=float(c.get("col_index", 100)),
                    score=float(tier.get("score", 0.5)),
                )
                p.cities[city.name.lower()] = city
        for c in loc.get("penalized", []):
            city = City(
                name=c["name"],
                state=c.get("state", ""),
                col_index=float(c.get("col_index", 100)),
                score=float(c.get("score", 0.3)),
            )
            p.cities[city.name.lower()] = city

        sig = raw.get("signals", {})
        p.green_signals = [Signal(s["phrase"].lower(), int(s.get("weight", 1)))
                           for s in sig.get("green", [])]
        p.red_signals = [Signal(s["phrase"].lower(), int(s.get("weight", 1)))
                         for s in sig.get("red", [])]
        return p

    # ------------------------------------------------------------- accessors

    @property
    def name(self) -> str:
        return self.identity.get("name", "the candidate")

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else "there"

    @property
    def seniority(self) -> dict[str, Any]:
        return self.raw.get("seniority", {})

    @property
    def compensation(self) -> dict[str, Any]:
        return self.raw.get("compensation", {})

    @property
    def location(self) -> dict[str, Any]:
        return self.raw.get("location", {})

    @property
    def industries(self) -> dict[str, Any]:
        return self.raw.get("industries", {})

    @property
    def hard_filters(self) -> dict[str, Any]:
        return self.raw.get("hard_filters", {})

    @property
    def weights(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw.get("weights", {}).items()}

    def city(self, name: str) -> City | None:
        """Look up a city by fuzzy name.

        Job APIs return location strings like 'Chicago, IL' or
        'Overland Park, Johnson County'. We match on any comma-separated part.
        """
        if not name:
            return None
        cleaned = name.lower()
        if cleaned in self.cities:
            return self.cities[cleaned]
        for part in re.split(r"[,/|]", cleaned):
            part = part.strip()
            if part in self.cities:
                return self.cities[part]
        # Last resort: substring containment, longest key first so that
        # 'Kansas City' wins over a hypothetical 'Kansas'.
        for key in sorted(self.cities, key=len, reverse=True):
            if key in cleaned:
                return self.cities[key]
        return None

    def top_families(self, n: int = 4) -> list[RoleFamily]:
        """The n most-wanted role families, used to fan out a broad search."""
        return sorted(self.role_families.values(), key=lambda f: -f.weight)[:n]

    def search_queries(self, n_families: int = 4, per_family: int = 1) -> list[str]:
        """Concrete keyword strings to send to the jobs API for a broad search."""
        out: list[str] = []
        for fam in self.top_families(n_families):
            out.extend(fam.queries[:per_family])
        return out

    def skills_by_tier(self, tier: str) -> list[str]:
        return [s.name for s in self.skills if s.tier == tier]

    # --------------------------------------------------------- prompt export

    def to_prompt_block(self) -> str:
        """A compact, complete rendering of the profile for the system prompt.

        Deliberately terse. This goes into every request, so every token here
        is paid for on every turn. It carries the facts the model needs to
        write commentary; the scoring engine does the actual ranking.
        """
        ident = self.identity
        comp = self.compensation
        sen = self.seniority
        loc = self.location

        tier1 = [c.name for c in self.cities.values() if c.score >= 0.95]
        tier2 = [c.name for c in self.cities.values() if 0.80 <= c.score < 0.95]

        lines = [
            f"NAME: {ident.get('name')}",
            f"HEADLINE: {ident.get('headline')}",
            f"EDUCATION: {ident.get('degree')}, {ident.get('school')}, graduating {ident.get('graduation')}",
            f"BASED IN: {ident.get('home_base')}",
            "",
            f"SENIORITY TARGET: {sen.get('target')} level, "
            f"max {sen.get('max_years_required')} years experience required.",
            "",
            "CORE SKILLS: " + ", ".join(self.skills_by_tier("core")),
            "WORKING SKILLS: " + ", ".join(self.skills_by_tier("working")),
            "FAMILIAR WITH: " + ", ".join(self.skills_by_tier("exposure")),
            "",
            "TARGET ROLES (in priority order): "
            + ", ".join(f.label for f in sorted(self.role_families.values(), key=lambda f: -f.weight)),
            "",
            f"TOP LOCATIONS: {', '.join(tier1)}",
            f"ALSO STRONGLY OPEN TO: {', '.join(tier2)}",
            f"REMOTE: {'yes, welcome anywhere in the US' if loc.get('remote_ok') else 'no'}. "
            f"RELOCATION: {'open' if loc.get('relocation_open') else 'not open'}.",
            "",
            f"COMPENSATION: floor ${comp.get('floor'):,}, target ${comp.get('target'):,}, "
            f"stretch ${comp.get('stretch'):,}. Salary is judged against local cost of living, "
            f"so a lower number in an affordable city can beat a higher number in an expensive one.",
            "",
            "PREFERRED INDUSTRIES: "
            + ", ".join(self.industries.get("preferred", {}).keys()),
            "",
            "WANTS: mentorship and structured onboarding, a data-driven culture where analysts "
            "influence decisions, a clear path to senior analyst, cross-team collaboration, "
            "mid-size to large employers.",
            "AVOIDS: purely administrative work, no growth path, fully in-office with zero "
            "flexibility, very small companies with no junior support structure.",
        ]
        return "\n".join(lines)


# =============================================================================
# Module-level singleton
# =============================================================================

_PROFILE: Profile | None = None


def get_profile(reload: bool = False) -> Profile:
    """Return the loaded profile, building it on first use."""
    global _PROFILE
    if _PROFILE is None or reload:
        _PROFILE = Profile.load()
    return _PROFILE


if __name__ == "__main__":
    prof = get_profile()
    print(prof.to_prompt_block())
    print()
    print("Search queries:", prof.search_queries())
    print("Weights sum:", sum(prof.weights.values()))
