from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ExpertRecommendation:
    """
    Explicit rule-based expert system outputs.
    """

    strategy: str
    focus_mode: str
    suggested_notes_type: str
    planning_advice: str
    reasoning: str


def recommend(
    *,
    days: int,
    hours_per_day: Optional[float],
    has_past_papers: bool,
) -> ExpertRecommendation:
    """
    Rule-based inference (Expert System).
    Keep rules simple and demonstrable for faculty evaluation.
    """
    rules_fired = []

    # Defaults
    strategy = "Full revision"
    focus_mode = "All selected topics"
    suggested_notes_type = "Full notes"
    planning_advice = "Start with high-priority topics, then revise once at the end."

    # Rules
    if days <= 3:
        strategy = "Crash revision"
        suggested_notes_type = "Short exam notes"
        planning_advice = "Do only high-priority topics first; keep the last half-day for rapid revision."
        rules_fired.append("days<=3 -> crash revision + short exam notes")
    elif 4 <= days <= 5:
        strategy = "Selective revision"
        suggested_notes_type = "Short exam notes"
        planning_advice = "Prioritize high-priority topics; cover remaining topics if time permits."
        rules_fired.append("4<=days<=5 -> selective revision")
    else:
        strategy = "Full revision"
        suggested_notes_type = "Full notes"
        rules_fired.append("days>5 -> full revision baseline")

    if hours_per_day is not None and hours_per_day > 0:
        if hours_per_day < 2:
            focus_mode = "Only high-priority topics"
            suggested_notes_type = "Short exam notes"
            planning_advice = "Keep study blocks small; do high-priority topics + quick definitions."
            rules_fired.append("hours/day<2 -> high-priority focus")
        elif hours_per_day > 4 and days > 5:
            focus_mode = "All selected topics (deep coverage)"
            suggested_notes_type = "Full notes"
            planning_advice = "Do full unit revision and add spaced revision every 2–3 days."
            rules_fired.append("hours/day>4 and days>5 -> full unit revision")

    if not has_past_papers:
        rules_fired.append("no past papers -> syllabus-based prioritization")

    reasoning = "Rules applied: " + (", ".join(rules_fired) if rules_fired else "baseline defaults")

    return ExpertRecommendation(
        strategy=strategy,
        focus_mode=focus_mode,
        suggested_notes_type=suggested_notes_type,
        planning_advice=planning_advice,
        reasoning=reasoning,
    )


def to_dict(r: ExpertRecommendation) -> Dict[str, Any]:
    return {
        "strategy": r.strategy,
        "focus_mode": r.focus_mode,
        "suggested_notes_type": r.suggested_notes_type,
        "planning_advice": r.planning_advice,
        "reasoning": r.reasoning,
    }

