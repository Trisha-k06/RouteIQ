from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class PlanDay:
    day: str
    topics: List[str]
    total_slots: int


def _slots_from_hours(hours_per_day: Optional[float]) -> int:
    """
    Convert hours/day into topic-block slots/day.
    We keep this student-friendly and stable for demos.
    """
    if hours_per_day is None or hours_per_day <= 0:
        return 3  # default "topic-block approach"
    # ~45 mins per topic block + quick recap
    return max(1, int(round(hours_per_day / 0.75)))


def make_plan(
    topic_importance: Dict[str, float],
    days: int,
    hours_per_day: Optional[float] = None,
) -> List[PlanDay]:
    """
    Day-wise study plan (Planning):
    - If hours/day provided, convert to topic blocks/day.
    - Otherwise use default topic-block approach.
    - Prioritize higher-importance topics first and sprinkle repetition lightly.
    """
    if days <= 0:
        return []

    slots_per_day = _slots_from_hours(hours_per_day)

    # Build weighted topic list: topic repeated by importance buckets (cap to avoid huge)
    weighted: List[str] = []
    for topic, score in sorted(topic_importance.items(), key=lambda x: x[1], reverse=True):
        # repeat 1..4 based on score, for a spaced-repetition feel without overfitting
        if score >= 2.3:
            repeat = 4
        elif score >= 1.9:
            repeat = 3
        elif score >= 1.4:
            repeat = 2
        else:
            repeat = 1
        weighted.extend([topic] * repeat)

    # If no data, return empty plan
    if not weighted:
        return [PlanDay(day=f"Day {i+1}", topics=[], total_slots=slots_per_day) for i in range(days)]

    plan: List[PlanDay] = []
    idx = 0

    for d in range(days):
        chosen = []
        used = set()
        for _ in range(slots_per_day):
            # pick next topic avoiding duplicates in same day if possible
            tries = 0
            while tries < len(weighted):
                t = weighted[idx % len(weighted)]
                idx += 1
                tries += 1
                if t not in used or len(used) >= len(weighted):
                    chosen.append(t)
                    used.add(t)
                    break
        plan.append(PlanDay(day=f"Day {d+1}", topics=chosen, total_slots=slots_per_day))

    return plan


def top_topics(topic_importance: Dict[str, float], k: int = 10) -> List[Tuple[str, float]]:
    return sorted(topic_importance.items(), key=lambda x: x[1], reverse=True)[:k]