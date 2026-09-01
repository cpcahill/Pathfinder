"""
Pathfinder UI

The design system: one stylesheet and a handful of HTML component renderers.

Design notes
------------
The previous interface was a blue gradient banner over default Streamlit
widgets, with a long tail of !important overrides fighting dark mode. This
replaces it with a dark-first system closer to how current developer tools
look: near-black canvas, a single elevated surface tone, hairline borders
instead of drop shadows, uppercase micro-labels, tabular figures for anything
numeric, and colour reserved almost entirely for meaning rather than decoration.

Colour carries exactly one job here: the fit band. Strong, good, fair and weak
each own a hue, and nothing else in the interface competes for that signal.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

BAND_COLORS = {
    "Strong": "#34D399",
    "Good":   "#38BDF8",
    "Fair":   "#FBBF24",
    "Weak":   "#FB7185",
}

STATUS_COLORS = {
    "Applied":      "#8B94A7",
    "Interviewing": "#38BDF8",
    "Offer":        "#34D399",
    "Rejected":     "#4B5163",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
  --bg:        #08090B;
  --surface:   #0E1014;
  --surface-2: #14171D;
  --line:      #1E222B;
  --line-soft: #171A21;
  --text:      #E8EAED;
  --text-dim:  #9BA3B4;
  --text-mute: #6B7385;
  --accent:    #8B5CF6;
  --radius:    12px;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

html, body, [class*="css"], .stApp { font-family: var(--sans); }
.stApp { background: var(--bg); }

#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
.block-container { padding: 1.4rem 2.2rem 4rem !important; max-width: 1320px; }

h1, h2, h3, h4 { font-family: var(--sans); letter-spacing: -0.02em; color: var(--text); }

.pf-eyebrow {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-mute); margin: 0 0 0.55rem;
}

.pf-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 0 1.1rem; margin-bottom: 1.4rem;
  border-bottom: 1px solid var(--line);
}
.pf-brand { display: flex; align-items: baseline; gap: 0.7rem; }
.pf-logo { font-size: 1.32rem; font-weight: 700; letter-spacing: -0.035em; color: var(--text); }
.pf-logo span {
  background: linear-gradient(92deg, #A78BFA 0%, #38BDF8 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.pf-tagline { font-size: 0.82rem; color: var(--text-mute); font-weight: 400; }
.pf-nav-right { display: flex; align-items: center; gap: 0.5rem; }

.pf-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-size: 0.72rem; font-weight: 500; color: var(--text-dim);
  background: var(--surface); border: 1px solid var(--line);
  padding: 0.3rem 0.66rem; border-radius: 999px; white-space: nowrap;
}
.pf-dot { width: 6px; height: 6px; border-radius: 50%; background: #34D399; }

.pf-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.6rem; }
.pf-stat {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 0.95rem 1.05rem;
  transition: border-color 0.16s ease;
}
.pf-stat:hover { border-color: #2A303C; }
.pf-stat-label {
  font-size: 0.66rem; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--text-mute); margin-bottom: 0.5rem;
}
.pf-stat-value {
  font-family: var(--mono); font-size: 1.7rem; font-weight: 600;
  color: var(--text); line-height: 1; font-variant-numeric: tabular-nums;
}
.pf-stat-sub { font-size: 0.73rem; color: var(--text-mute); margin-top: 0.42rem; }
.pf-stat-sub b { color: var(--text-dim); font-weight: 500; }

.pf-card {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1.05rem 1.15rem;
  margin-bottom: 0.7rem; transition: border-color 0.16s ease, background 0.16s ease;
}
.pf-card:hover { border-color: #2C3340; background: #0F1116; }
.pf-card-top { display: flex; gap: 1rem; align-items: flex-start; }

.pf-ring { flex: 0 0 auto; position: relative; width: 54px; height: 54px; }
.pf-ring-num {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 1.02rem; font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.pf-card-body { flex: 1 1 auto; min-width: 0; }
.pf-card-title {
  font-size: 1.0rem; font-weight: 600; color: var(--text);
  letter-spacing: -0.012em; line-height: 1.3; margin: 0 0 0.2rem;
}
.pf-card-org { font-size: 0.85rem; color: var(--text-dim); margin-bottom: 0.6rem; }
.pf-card-org b { color: var(--text); font-weight: 500; }

.pf-chips { display: flex; flex-wrap: wrap; gap: 0.34rem; margin-bottom: 0.7rem; }
.pf-chip {
  font-size: 0.7rem; font-weight: 500; padding: 0.19rem 0.5rem;
  border-radius: 5px; border: 1px solid var(--line);
  background: var(--surface-2); color: var(--text-dim);
}
.pf-chip.warn { border-color: #4A3A1C; background: #211A0E; color: #E9B949; }
.pf-chip.good { border-color: #1B4436; background: #0D211A; color: #34D399; }

.pf-reasons { list-style: none; padding: 0; margin: 0 0 0.55rem; }
.pf-reasons li {
  font-size: 0.815rem; color: var(--text-dim); line-height: 1.5;
  padding-left: 1rem; position: relative;
}
.pf-reasons li::before {
  content: ""; position: absolute; left: 0; top: 0.58em;
  width: 5px; height: 5px; border-radius: 50%; background: #34D399;
}
.pf-reasons li.minus::before { background: #6B7385; }

.pf-card-foot {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; margin-top: 0.75rem; padding-top: 0.75rem;
  border-top: 1px solid var(--line-soft);
}
.pf-apply {
  font-size: 0.78rem; font-weight: 600; color: #C4B5FD !important;
  text-decoration: none !important; border: 1px solid #332A55;
  background: #171331; padding: 0.36rem 0.85rem; border-radius: 7px;
  transition: all 0.15s ease; white-space: nowrap;
}
.pf-apply:hover { background: #201A45; border-color: #4A3D7A; color: #DDD6FE !important; }

.pf-why { margin-top: 0.2rem; }
.pf-why summary {
  cursor: pointer; list-style: none; font-size: 0.74rem; font-weight: 500;
  color: var(--text-mute); user-select: none; padding: 0.1rem 0;
}
.pf-why summary::-webkit-details-marker { display: none; }
.pf-why summary:hover { color: var(--text-dim); }
.pf-why summary::before { content: "> "; font-size: 0.7rem; }
.pf-why[open] summary::before { content: "v "; }

.pf-axis { display: grid; grid-template-columns: 118px 78px 1fr; gap: 0.6rem;
           align-items: center; padding: 0.28rem 0; }
.pf-axis-name { font-size: 0.74rem; color: var(--text-dim); }
.pf-axis-track { height: 4px; background: #1B1F27; border-radius: 2px; overflow: hidden; }
.pf-axis-fill { height: 100%; border-radius: 2px; }
.pf-axis-detail { font-size: 0.72rem; color: var(--text-mute); line-height: 1.4; }
.pf-axis-pts { font-family: var(--mono); font-size: 0.7rem; color: var(--text-mute);
               font-variant-numeric: tabular-nums; }

.pf-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.pf-table th {
  text-align: left; font-size: 0.66rem; font-weight: 600; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--text-mute);
  padding: 0 0.7rem 0.55rem; border-bottom: 1px solid var(--line);
}
.pf-table td { padding: 0.62rem 0.7rem; border-bottom: 1px solid var(--line-soft); color: var(--text-dim); }
.pf-table tr:last-child td { border-bottom: none; }
.pf-table td.name { color: var(--text); font-weight: 500; }
.pf-status { display: inline-flex; align-items: center; gap: 0.36rem;
             font-size: 0.74rem; font-weight: 500; }
.pf-status i { width: 6px; height: 6px; border-radius: 50%; display: inline-block; font-style: normal; }

.pf-panel {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1.1rem 1.2rem; margin-bottom: 1rem;
}
.pf-empty {
  border: 1px dashed var(--line); border-radius: var(--radius);
  padding: 2.2rem 1.4rem; text-align: center; color: var(--text-mute);
  font-size: 0.86rem; background: #0B0D10;
}

[data-testid="stChatMessage"] {
  background: var(--surface) !important; border: 1px solid var(--line);
  border-radius: var(--radius); padding: 0.85rem 1.05rem; margin-bottom: 0.6rem;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li { color: var(--text-dim); font-size: 0.885rem; line-height: 1.62; }
[data-testid="stChatMessage"] strong { color: var(--text); font-weight: 600; }
[data-testid="stChatMessage"] a { color: #A78BFA; }
[data-testid="stChatMessage"] code {
  font-family: var(--mono); font-size: 0.82em; color: #A78BFA;
  background: var(--surface-2); padding: 0.1rem 0.35rem; border-radius: 4px;
}
[data-testid="stChatMessage"] table { border-collapse: collapse; font-size: 0.8rem; }
[data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td {
  border: 1px solid var(--line); padding: 0.4rem 0.6rem;
}

/* Streamlit ships bright Material avatars that fight this palette. Tone them
   down to match, and distinguish the two speakers with the bubble itself
   rather than with colour. :has() lets us style the parent by its avatar. */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  width: 26px !important; height: 26px !important;
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  color: var(--text-mute) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: #171331 !important; border-color: #332A55 !important;
  color: #A78BFA !important;
}
[data-testid="stChatMessage"] [data-testid="stIconMaterial"] { font-size: 15px !important; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: transparent !important; border-style: dashed;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
  color: var(--text) !important; font-weight: 500;
}

[data-testid="stChatInput"] textarea { font-family: var(--sans) !important; font-size: 0.9rem !important; }
[data-testid="stChatInput"] > div { border-radius: 10px !important; border-color: var(--line) !important; }

.stButton > button {
  background: var(--surface); color: var(--text-dim);
  border: 1px solid var(--line); border-radius: 9px;
  font-size: 0.8rem; font-weight: 500; padding: 0.5rem 0.8rem;
  transition: all 0.15s ease; width: 100%;
}
.stButton > button:hover {
  background: var(--surface-2); border-color: #333B4A;
  color: var(--text); transform: translateY(-1px);
}
.stButton > button:focus { box-shadow: none !important; color: var(--text) !important; }

section[data-testid="stSidebar"] { background: #0B0C0F; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

[data-testid="stExpander"] {
  border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
  background: var(--surface) !important;
}
[data-testid="stExpander"] summary { font-size: 0.84rem !important; color: var(--text-dim) !important; }

div[data-testid="stSpinner"] p { color: var(--text-mute); font-size: 0.82rem; }
hr { border-color: var(--line) !important; margin: 1.2rem 0 !important; }

@media (max-width: 900px) {
  .pf-stats { grid-template-columns: repeat(2, 1fr); }
  .block-container { padding: 1rem 1rem 3rem !important; }
  .pf-nav { flex-direction: column; align-items: flex-start; gap: 0.7rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# Components
# =============================================================================

def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="pf-eyebrow">{_esc(text)}</div>', unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.markdown(f'<div class="pf-empty">{_esc(message)}</div>', unsafe_allow_html=True)


def nav(pills: list[tuple[str, bool]]) -> None:
    """Top bar. Each pill is (label, show_live_dot)."""
    pill_html = ""
    for label, live in pills:
        dot = '<span class="pf-dot"></span>' if live else ""
        pill_html += f'<span class="pf-pill">{dot}{_esc(label)}</span>'
    st.markdown(
        '<div class="pf-nav">'
        '  <div class="pf-brand">'
        '    <div class="pf-logo">Path<span>finder</span></div>'
        '    <div class="pf-tagline">Scored job search</div>'
        '  </div>'
        f' <div class="pf-nav-right">{pill_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def stat_row(stats: list[tuple[str, Any, str]]) -> None:
    """Four KPI tiles. Each stat is (label, value, sub_caption_html)."""
    cells = ""
    for label, value, sub in stats:
        cells += (
            '<div class="pf-stat">'
            f'  <div class="pf-stat-label">{_esc(label)}</div>'
            f'  <div class="pf-stat-value">{_esc(value)}</div>'
            f'  <div class="pf-stat-sub">{sub}</div>'
            '</div>'
        )
    st.markdown(f'<div class="pf-stats">{cells}</div>', unsafe_allow_html=True)


def _ring(score: int, color: str) -> str:
    """A 54px progress ring rendered as inline SVG."""
    radius = 22
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - max(0, min(100, score)) / 100)
    return (
        '<div class="pf-ring">'
        '<svg width="54" height="54" viewBox="0 0 54 54">'
        f'<circle cx="27" cy="27" r="{radius}" fill="none" stroke="#1B1F27" stroke-width="4"/>'
        f'<circle cx="27" cy="27" r="{radius}" fill="none" stroke="{color}" stroke-width="4"'
        f' stroke-linecap="round" stroke-dasharray="{circumference:.1f}"'
        f' stroke-dashoffset="{offset:.1f}" transform="rotate(-90 27 27)"/>'
        '</svg>'
        f'<div class="pf-ring-num" style="color:{color}">{score}</div>'
        '</div>'
    )


def job_card(fit) -> None:
    """One scored listing, with an expandable per-axis breakdown."""
    job = fit.job
    color = BAND_COLORS.get(fit.band, "#8B94A7")

    chips = f'<span class="pf-chip">{_esc(job.location or "Location not stated")}</span>'
    if fit.family_label:
        chips += f'<span class="pf-chip">{_esc(fit.family_label)}</span>'
    location_text = (job.location or "").lower()
    for flag in fit.flags:
        # The location chip already reads "Remote" for these, so a second
        # Remote chip beside it is noise.
        if flag == "Remote" and "remote" in location_text:
            continue
        cls = "good" if flag == "Remote" else "warn"
        chips += f'<span class="pf-chip {cls}">{_esc(flag)}</span>'

    reasons = ""
    for reason in fit.top_reasons(3):
        reasons += f"<li>{_esc(reason)}</li>"
    for weak in fit.weak_spots(1):
        reasons += f'<li class="minus">{_esc(weak)}</li>'

    axes = ""
    for axis in fit.axes:
        pct = int(axis.raw * 100)
        axes += (
            '<div class="pf-axis">'
            f'  <div class="pf-axis-name">{_esc(axis.label)}</div>'
            '  <div><div class="pf-axis-track">'
            f'    <div class="pf-axis-fill" style="width:{pct}%;background:{color}"></div>'
            '  </div></div>'
            '  <div class="pf-axis-detail">'
            f'    <span class="pf-axis-pts">{axis.points:.0f}/{axis.weight:.0f}</span> '
            f'    {_esc(axis.detail)}</div>'
            '</div>'
        )
    for label, delta in fit.adjustments:
        axes += (
            '<div class="pf-axis"><div class="pf-axis-name">Adjustment</div><div></div>'
            f'<div class="pf-axis-detail"><span class="pf-axis-pts">{delta:+.0f}</span> '
            f'{_esc(label)}</div></div>'
        )

    posted = f"Posted {job.created}" if job.created else "Date unknown"
    apply_btn = ""
    if job.url:
        apply_btn = (f'<a class="pf-apply" href="{_esc(job.url)}" target="_blank" '
                     f'rel="noopener">Open listing</a>')

    st.markdown(
        '<div class="pf-card">'
        '  <div class="pf-card-top">'
        f'   {_ring(fit.score, color)}'
        '    <div class="pf-card-body">'
        f'      <div class="pf-card-title">{_esc(job.title)}</div>'
        f'      <div class="pf-card-org"><b>{_esc(job.company or "Company undisclosed")}</b>'
        f'        &nbsp;&middot;&nbsp; <span style="color:{color}">{fit.band} fit</span></div>'
        f'      <div class="pf-chips">{chips}</div>'
        f'      <ul class="pf-reasons">{reasons}</ul>'
        f'      <details class="pf-why"><summary>Score breakdown</summary>{axes}</details>'
        '      <div class="pf-card-foot">'
        f'        <span style="font-size:0.73rem;color:var(--text-mute)">{_esc(posted)}</span>'
        f'        {apply_btn}'
        '      </div>'
        '    </div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )


def pipeline_table(apps: list[dict[str, Any]], limit: int = 8) -> None:
    rows = ""
    for app in apps[:limit]:
        status = app.get("status", "Applied")
        color = STATUS_COLORS.get(status, "#8B94A7")
        score = app.get("fit_score")
        rows += (
            '<tr>'
            f'  <td class="name">{_esc(app.get("company"))}</td>'
            f'  <td>{_esc(app.get("role"))}</td>'
            f'  <td style="font-family:var(--mono);font-size:0.78rem">{_esc(app.get("date_applied"))}</td>'
            f'  <td style="font-family:var(--mono);font-size:0.78rem">{score if score else "&mdash;"}</td>'
            f'  <td><span class="pf-status" style="color:{color}">'
            f'    <i style="background:{color}"></i>{_esc(status)}</span></td>'
            '</tr>'
        )
    st.markdown(
        '<table class="pf-table">'
        '<thead><tr><th>Company</th><th>Role</th><th>Applied</th>'
        '<th>Fit</th><th>Status</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def funnel_bar(counts: dict[str, int]) -> None:
    """A single proportional bar of the pipeline instead of a bar chart.

    With a handful of applications a full chart is more furniture than signal.
    One bar reads instantly and takes a fifth of the vertical space.
    """
    order = ("Applied", "Interviewing", "Offer", "Rejected")
    total = sum(counts.values()) or 1
    segments = ""
    for status in order:
        n = counts.get(status, 0)
        if n:
            segments += (f'<div style="width:{n / total * 100:.1f}%;'
                         f'background:{STATUS_COLORS[status]}" title="{status}: {n}"></div>')
    legend = ""
    for status in order:
        legend += (
            f'<span class="pf-status" style="color:{STATUS_COLORS[status]};margin-right:0.9rem">'
            f'<i style="background:{STATUS_COLORS[status]}"></i>{status} '
            f'<span style="font-family:var(--mono);color:var(--text-mute)">'
            f'{counts.get(status, 0)}</span></span>'
        )
    st.markdown(
        '<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;'
        f'background:#1B1F27;margin-bottom:0.7rem">{segments}</div>'
        f'<div style="display:flex;flex-wrap:wrap">{legend}</div>',
        unsafe_allow_html=True,
    )
