"""
Pathfinder Scoring Engine

Turns a raw job posting into an explainable 0-100 fit score.

Why this exists
---------------
The original Pathfinder handed ten raw listings to an LLM and asked it to
eyeball which ones fit. That works, in the sense that it produces sentences,
but it is not precise and it is not reproducible: the same listing could be
called a strong fit on one turn and skipped on the next, and nothing could be
audited afterwards.

This module does the ranking deterministically. Every listing is run through
hard filters first (things that make a posting simply wrong: senior titles,
clearance requirements, internships), then scored on seven weighted axes. The
LLM still writes the commentary, which is what it is genuinely good at, but it
no longer decides the ordering.

Every score carries its own breakdown, so the UI can show exactly why a job
ranked where it did, and the weights that produced it live in profile.yaml
where they can be tuned without touching code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from pathfinder_profile import Profile, get_profile


# =============================================================================
# Normalized job record
# =============================================================================

@dataclass
class Job:
    """A job posting, normalized away from any single provider's schema."""
    id: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    description: str = ""
    url: str = ""
    created: str = ""                 # YYYY-MM-DD
    salary_min: float | None = None
    salary_max: float | None = None
    salary_is_predicted: bool = False
    contract_time: str = ""           # full_time | part_time
    contract_type: str = ""           # permanent | contract
    category: str = ""
    source: str = "adzuna"

    @property
    def blob(self) -> str:
        """Everything searchable about this posting, lowercased."""
        return " ".join([
            self.title, self.company, self.location,
            self.category, self.description,
        ]).lower()

    @property
    def salary_point(self) -> float | None:
        """A single representative salary number."""
        if self.salary_min and self.salary_max:
            return (self.salary_min + self.salary_max) / 2
        return self.salary_min or self.salary_max or None

    @property
    def age_days(self) -> int | None:
        if not self.created:
            return None
        try:
            posted = datetime.strptime(self.created[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
        return max((date.today() - posted).days, 0)

    @classmethod
    def from_adzuna(cls, raw: dict[str, Any]) -> "Job":
        return cls(
            id=str(raw.get("id", "")),
            title=raw.get("title", "") or "",
            company=(raw.get("company") or {}).get("display_name", "") or "",
            location=(raw.get("location") or {}).get("display_name", "") or "",
            description=(raw.get("description") or "").replace("\n", " ").strip(),
            url=raw.get("redirect_url", "") or "",
            created=(raw.get("created") or "")[:10],
            salary_min=raw.get("salary_min"),
            salary_max=raw.get("salary_max"),
            salary_is_predicted=bool(raw.get("salary_is_predicted") in ("1", 1, True)),
            contract_time=raw.get("contract_time", "") or "",
            contract_type=raw.get("contract_type", "") or "",
            category=(raw.get("category") or {}).get("label", "") or "",
        )

    def dedupe_key(self) -> str:
        """Identity for deduplication across multiple search queries.

        Fanning out across role families means the same posting comes back
        more than once. Company plus normalized title is a better key than the
        provider's id, because aggregators often list the same job twice under
        different ids.
        """
        t = re.sub(r"[^a-z0-9 ]", "", self.title.lower())
        t = re.sub(r"\s+", " ", t).strip()
        c = re.sub(r"[^a-z0-9]", "", self.company.lower())
        return f"{c}|{t}"


# =============================================================================
# Result objects
# =============================================================================

@dataclass
class Axis:
    key: str
    label: str
    raw: float                        # 0..1
    weight: float                     # points available
    detail: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def points(self) -> float:
        return self.raw * self.weight


@dataclass
class Fit:
    job: Job
    axes: list[Axis] = field(default_factory=list)
    adjustments: list[tuple[str, float]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    family_label: str = ""
    rejected: bool = False
    reject_reason: str = ""

    @property
    def score(self) -> int:
        base = sum(a.points for a in self.axes)
        base += sum(delta for _, delta in self.adjustments)
        return int(round(max(0.0, min(100.0, base))))

    @property
    def band(self) -> str:
        s = self.score
        if s >= 80:
            return "Strong"
        if s >= 66:
            return "Good"
        if s >= 50:
            return "Fair"
        return "Weak"

    def top_reasons(self, n: int = 3) -> list[str]:
        """The axes contributing the most, phrased for a human."""
        ranked = sorted(self.axes, key=lambda a: -a.points)
        return [a.detail for a in ranked[:n] if a.detail]

    def weak_spots(self, n: int = 2) -> list[str]:
        """Axes that lost the most available points."""
        ranked = sorted(self.axes, key=lambda a: a.raw)
        return [a.detail for a in ranked[:n] if a.raw < 0.5 and a.detail]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.job.title,
            "company": self.job.company,
            "location": self.job.location,
            "url": self.job.url,
            "created": self.job.created,
            "score": self.score,
            "band": self.band,
            "family": self.family_label,
            "flags": self.flags,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "axes": [
                {"label": a.label, "raw": round(a.raw, 3),
                 "points": round(a.points, 1), "weight": a.weight,
                 "detail": a.detail}
                for a in self.axes
            ],
            "adjustments": self.adjustments,
        }


# =============================================================================
# Helpers
# =============================================================================

def _has_token(text: str, token: str) -> bool:
    """Word-boundary containment. 'lead' must not match 'leadership'."""
    return re.search(rf"\b{re.escape(token.lower())}\b", text) is not None


_YEARS_RE = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:-|–|to)?\s*\d{0,2}\s*\+?\s*year", re.IGNORECASE
)


def required_years(description: str) -> int | None:
    """Smallest number of years of experience the posting appears to demand.

    We only count a number if the word 'experience' shows up shortly after it,
    which avoids matching things like 'for over 40 years we have served'.
    """
    best: int | None = None
    for m in _YEARS_RE.finditer(description):
        window = description[m.end(): m.end() + 60].lower()
        if "experien" not in window:
            continue
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if n > 25:                     # company age, not a requirement
            continue
        best = n if best is None else min(best, n)
    return best


def _is_remote(job: Job) -> bool:
    text = f"{job.title} {job.location} {job.description}".lower()
    if _has_token(text, "remote") or "work from home" in text or "wfh" in text:
        # 'no remote' / 'not remote' should not count as remote
        if re.search(r"\b(no|not|non)[ -]remote\b", text):
            return False
        return True
    return False


# =============================================================================
# Hard filters
# =============================================================================

def screen(job: Job, profile: Profile) -> str | None:
    """Return a rejection reason, or None if the posting survives screening."""
    hf = profile.hard_filters
    sen = profile.seniority
    title = job.title.lower()
    desc = job.description.lower()

    for token in hf.get("drop_if_title_contains", []):
        if token.lower() in title:
            return f"Title contains '{token}'"

    for token in sen.get("excluded_title_tokens", []):
        if _has_token(title, str(token)):
            return f"Seniority mismatch: title says '{token}'"

    for phrase in hf.get("drop_if_description_contains", []):
        if phrase.lower() in desc:
            return f"Posting requires '{phrase}'"

    yrs = required_years(job.description)
    cap = sen.get("max_years_required")
    if yrs is not None and cap is not None and yrs > int(cap):
        return f"Requires {yrs}+ years experience (cap is {cap})"

    return None


# =============================================================================
# Individual axes
# =============================================================================

def _score_role_fit(job: Job, profile: Profile) -> tuple[float, str, str]:
    """Match the posting's title to the candidate's wanted role families."""
    title = job.title.lower()
    best_weight, best_label = 0.0, ""

    for fam in profile.role_families.values():
        for token in fam.titles:
            if token.strip() and token in title:
                if fam.weight > best_weight:
                    best_weight, best_label = fam.weight, fam.label

    if best_weight:
        # A new-grad-flavoured title on top of a wanted family is the ideal case.
        for token in profile.seniority.get("preferred_title_tokens", []):
            if _has_token(title, str(token)):
                return (min(1.0, best_weight + 0.08), best_label,
                        f"{best_label} role, explicitly early-career")
        return best_weight, best_label, f"{best_label} role, a top target"

    # Nothing matched a family, but the title still smells analytical.
    for generic, val in (("analyst", 0.45), ("analytics", 0.45),
                         ("engineer", 0.35), ("data", 0.35),
                         ("consultant", 0.35), ("developer", 0.30)):
        if generic in title:
            return val, "Adjacent", "Adjacent title, not a named target role"

    return 0.12, "Off-target", "Title sits outside the target role families"


def _score_skills(job: Job, profile: Profile) -> tuple[float, str, list[str], list[str]]:
    """How much of the candidate's toolkit this posting actually asks for.

    Note on saturation: a job description that names three core skills is a
    complete match for scoring purposes. Requiring a posting to mention every
    skill would punish short descriptions, and many job APIs truncate.
    """
    blob = job.blob
    matched: list[str] = []
    value = 0.0

    for skill in profile.skills:
        if any(alias in blob for alias in skill.aliases):
            matched.append(skill.name)
            value += skill.value

    missing = [s.name for s in profile.skills
               if s.tier == "core" and s.name not in matched]

    raw = min(1.0, value / 3.0)        # 3.0 == roughly three core skills
    if matched:
        detail = "Asks for " + ", ".join(matched[:4])
    else:
        detail = "Description names none of the candidate's skills"
    return raw, detail, matched, missing


def _score_location(job: Job, profile: Profile) -> tuple[float, str, float]:
    """Returns (score, detail, col_index) so compensation can reuse the index."""
    loc = profile.location

    if _is_remote(job):
        return float(loc.get("remote_score", 0.9)), "Remote friendly", 100.0

    city = profile.city(job.location)
    if city:
        if city.score >= 0.95:
            phrasing = "top-choice city"
        elif city.score >= 0.80:
            phrasing = "a city they would happily relocate to"
        elif city.score >= 0.60:
            phrasing = "an acceptable market"
        else:
            phrasing = "a market they are lukewarm on"
        return city.score, f"{city.name} is {phrasing}", city.col_index

    fallback = float(loc.get("unlisted_score", 0.45))
    where = job.location or "location not stated"
    return fallback, f"{where} is outside the listed preferences", 100.0


def _score_compensation(job: Job, profile: Profile,
                        col_index: float) -> tuple[float, str]:
    """Compare pay to the target, normalized for local cost of living."""
    comp = profile.compensation
    floor = float(comp.get("floor", 55000))
    target = float(comp.get("target", 75000))
    stretch = float(comp.get("stretch", target * 1.3))
    floor_score = float(comp.get("floor_score", 0.40))
    target_score = float(comp.get("target_score", 0.85))

    point = job.salary_point
    if not point:
        return float(comp.get("unknown_score", 0.55)), "No salary published"

    adjusted = point / (col_index / 100.0)

    # The axis tops out at the stretch number, not at twice the target. The
    # previous curve reached 1.0 only at 2x target, which for an entry-level
    # search meant the compensation axis could never be fully earned: even a
    # posting well above the target scored ~0.9. Anchoring the top to stretch
    # makes the band that actually matters, target to stretch, the part that
    # moves the score.
    if adjusted >= stretch:
        raw = 1.0
        verdict = "at or above the stretch number"
    elif adjusted >= target:
        raw = target_score + (1.0 - target_score) * (adjusted - target) / max(stretch - target, 1)
        verdict = "above target"
    elif adjusted >= floor:
        raw = floor_score + (target_score - floor_score) * (adjusted - floor) / max(target - floor, 1)
        verdict = "inside the acceptable band"
    else:
        raw = max(0.0, floor_score * adjusted / max(floor, 1))
        verdict = "below the floor"

    note = f"${point:,.0f} listed"
    if abs(col_index - 100) > 3:
        note += f", ${adjusted:,.0f} adjusted for cost of living"
    if job.salary_is_predicted:
        note += " (estimated by the job board)"
    return raw, f"{note}, {verdict}"


def _score_industry(job: Job, profile: Profile) -> tuple[float, str]:
    blob = job.blob
    prefs = profile.industries.get("preferred", {})
    hits: list[str] = []
    for industry, keywords in prefs.items():
        if any(k.lower() in blob for k in keywords):
            hits.append(industry.replace("_", " "))
    if hits:
        return 1.0, "Preferred industry: " + ", ".join(hits[:2])
    return float(profile.industries.get("neutral_score", 0.55)), "Industry not a stated preference"


def _score_signals(job: Job, profile: Profile) -> tuple[float, str, list[str]]:
    """Green and red flag phrases drawn from the job-search strategy guide."""
    blob = job.blob
    green_hits, red_hits = [], []
    green_pts = red_pts = 0

    for sig in profile.green_signals:
        if sig.phrase in blob:
            green_hits.append(sig.phrase)
            green_pts += sig.weight
    for sig in profile.red_signals:
        if sig.phrase in blob:
            red_hits.append(sig.phrase)
            red_pts += sig.weight

    raw = max(0.0, min(1.0, 0.5 + green_pts / 12.0 - red_pts / 8.0))

    if green_hits and not red_hits:
        detail = "Green flags: " + ", ".join(green_hits[:3])
    elif red_hits and not green_hits:
        detail = "Red flags: " + ", ".join(red_hits[:3])
    elif green_hits and red_hits:
        detail = f"Mixed signals: {green_hits[0]}, but also {red_hits[0]}"
    else:
        detail = "No strong signals either way"
    return raw, detail, red_hits


def _score_freshness(job: Job) -> tuple[float, str]:
    age = job.age_days
    if age is None:
        return 0.5, "Posting date unknown"
    if age <= 3:
        return 1.0, f"Posted {age} day{'s' if age != 1 else ''} ago"
    if age <= 30:
        raw = 1.0 - 0.7 * (age - 3) / 27.0
        return raw, f"Posted {age} days ago"
    return 0.2, f"Posted {age} days ago, likely stale"


# =============================================================================
# The public entry point
# =============================================================================

def score_job(job: Job, profile: Profile | None = None) -> Fit:
    """Score a single posting. Rejected postings come back with rejected=True."""
    profile = profile or get_profile()
    w = profile.weights
    fit = Fit(job=job)

    reason = screen(job, profile)
    if reason:
        fit.rejected = True
        fit.reject_reason = reason
        return fit

    role_raw, family, role_detail = _score_role_fit(job, profile)
    skills_raw, skills_detail, matched, missing = _score_skills(job, profile)
    loc_raw, loc_detail, col_index = _score_location(job, profile)
    comp_raw, comp_detail = _score_compensation(job, profile, col_index)
    ind_raw, ind_detail = _score_industry(job, profile)
    sig_raw, sig_detail, red_hits = _score_signals(job, profile)
    fresh_raw, fresh_detail = _score_freshness(job)

    fit.family_label = family
    fit.matched_skills = matched
    fit.missing_skills = missing
    fit.axes = [
        Axis("role_fit", "Role fit", role_raw, w.get("role_fit", 28), role_detail),
        Axis("skills", "Skills overlap", skills_raw, w.get("skills", 24), skills_detail),
        Axis("location", "Location", loc_raw, w.get("location", 15), loc_detail),
        Axis("compensation", "Compensation", comp_raw, w.get("compensation", 12), comp_detail),
        Axis("industry", "Industry", ind_raw, w.get("industry", 8), ind_detail),
        Axis("signals", "Job quality signals", sig_raw, w.get("signals", 8), sig_detail),
        Axis("freshness", "Freshness", fresh_raw, w.get("freshness", 5), fresh_detail),
    ]

    # ---- flags and point adjustments -------------------------------------
    company_blob = f"{job.company} {job.category}".lower()
    for marker in profile.hard_filters.get("staffing_agency_markers", []):
        if marker.lower() in company_blob:
            fit.flags.append("Staffing agency")
            fit.adjustments.append(("Listed through a staffing agency", -4.0))
            break

    if job.contract_type == "contract":
        fit.flags.append("Contract")
        fit.adjustments.append(("Contract rather than permanent", -5.0))
    if job.contract_time == "part_time":
        fit.flags.append("Part time")
        fit.adjustments.append(("Part time", -8.0))
    if _is_remote(job):
        fit.flags.append("Remote")
    if not job.salary_point:
        fit.flags.append("No salary listed")
    if red_hits:
        fit.flags.append("Red flags")

    return fit


def score_and_rank(jobs: list[Job], profile: Profile | None = None,
                   limit: int | None = None) -> tuple[list[Fit], list[Fit]]:
    """Score every posting, dedupe, and split into kept and screened-out.

    Returns (ranked_keepers, rejected). Deduplication happens before scoring
    so the same posting arriving from three different queries costs one score
    and appears once.
    """
    profile = profile or get_profile()

    seen: set[str] = set()
    unique: list[Job] = []
    for job in jobs:
        key = job.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)

    kept, rejected = [], []
    for job in unique:
        fit = score_job(job, profile)
        (rejected if fit.rejected else kept).append(fit)

    kept.sort(key=lambda f: -f.score)
    if limit:
        kept = kept[:limit]
    return kept, rejected
