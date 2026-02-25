from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import date, timedelta


@dataclass
class PlanDay:
    day: str
    topics: List[str]
    total_slots: int


def make_plan(
    topic_frequency: Dict[str, int],
    days: int,
    slots_per_day: int = 3,
) -> List[PlanDay]:
    """
    slots_per_day = number of topic blocks/day (not hours). Simple & demo-friendly.
    We allocate high-frequency topics more slots (spaced repetition feel).
    """
    if days <= 0:
        return []

    # Build weighted topic list: topic repeated by frequency (cap to avoid huge)
    weighted: List[str] = []
    for topic, f in sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True):
        repeat = min(max(f, 1), 5)  # cap repeats
        weighted.extend([topic] * repeat)

    # If no freq data, return empty plan
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


def top_topics(topic_frequency: Dict[str, int], k: int = 10) -> List[Tuple[str, int]]:
    return sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:k]