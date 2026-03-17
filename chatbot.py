import re
from typing import Any, Dict, Optional, List, Tuple

from planner import top_topics, make_plan


def answer_user(
    message: str,
    analysis: Optional[Dict[str, Any]],
    days: int,
    hours_per_day: Optional[float],
) -> str:
    msg = message.lower().strip()

    if analysis is None:
        return "Upload syllabus and click Run Analysis first. Past papers are optional."

    importance = analysis.get("topic_importance", {})
    topics_by_unit = analysis.get("topics_by_unit", {})

    def _units_list() -> List[str]:
        return list(topics_by_unit.keys())

    def _all_topics_flat() -> List[str]:
        out = []
        for unit, topics in topics_by_unit.items():
            for tp in topics:
                out.append(f"{unit}: {tp}")
        return out

    # 1) Units listing
    if ("unit" in msg or "units" in msg) and ("show" in msg or "list" in msg):
        units = _units_list()
        if not units:
            return "I couldn't detect any units from the syllabus. Try pasting syllabus text instead of PDF."
        return "Units detected:\n" + "\n".join([f"- {u}" for u in units])

    # 1b) What topics were extracted?
    if ("topics" in msg and ("extracted" in msg or "found" in msg or "detected" in msg)) or ("what" in msg and "topics" in msg):
        lines = ["Extracted topics by unit:"]
        if not topics_by_unit:
            return "I couldn't extract topics. Paste syllabus text for best results."
        for unit, topics in topics_by_unit.items():
            lines.append(f"- {unit}:")
            for tp in topics[:18]:
                lines.append(f"  - {tp}")
            if len(topics) > 18:
                lines.append(f"  - ... (+{len(topics) - 18} more)")
        return "\n".join(lines)

    # 2) Past papers status
    if "past paper" in msg or "past papers" in msg:
        if analysis.get("has_past_papers"):
            return "Yes — I used past papers to rank topics by frequency."
        return "No past papers were uploaded. I’m ranking topics using syllabus topics only."

    # 3) Top topics / high-yield
    if "top" in msg and ("topic" in msg or "high yield" in msg or "important" in msg):
        tt = top_topics(importance, k=10)
        if not tt:
            return "I couldn't rank topics yet. Please run analysis again (and optionally upload past papers)."
        lines = ["Here are the top high-yield topics:"]
        for i, (t, score) in enumerate(tt, 1):
            lines.append(f"{i}. {t}  (score {score:.2f})")
        return "\n".join(lines)

    # 4) Plan request
    if "plan" in msg or "schedule" in msg or "timetable" in msg:
        # allow "make a 7 day plan" overrides
        m = re.search(r"(\d+)\s*[- ]*day", msg)
        req_days = int(m.group(1)) if m else days

        plan = make_plan(importance, days=req_days, hours_per_day=hours_per_day)
        slots = plan[0].total_slots if plan else 0
        lines = [f"Here’s a {req_days}-day plan ({slots} topic blocks/day):"]
        for pd in plan:
            lines.append(f"- {pd.day}: " + (" | ".join(pd.topics) if pd.topics else "(no topics found)"))
        return "\n".join(lines)

    # 5) Syllabus summary
    if "syllabus" in msg and ("summary" in msg or "overview" in msg or "units" in msg):
        lines = ["Here’s what I extracted from your syllabus:"]
        for unit, topics in topics_by_unit.items():
            lines.append(f"- {unit}: {len(topics)} topics")
        lines.append("Try: “show units”, “what topics were extracted”, “show top topics”, “make a 7 day plan”.")
        return "\n".join(lines)

    # 6) Why important?
    if "why" in msg and ("important" in msg or "priority" in msg):
        # Find best match among known topics
        candidates: List[Tuple[str, float]] = []
        for full_topic, score in importance.items():
            name = full_topic.split(":", 1)[-1].strip().lower()
            if name and name in msg:
                candidates.append((full_topic, score))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            t, score = candidates[0]
            if analysis.get("has_past_papers"):
                return f"'{t}' is high-priority because it matched past-paper questions frequently (importance score {score:.2f})."
            return f"'{t}' is prioritized using syllabus-only reasoning (importance score {score:.2f})."

        return "Tell me the topic name (or copy-paste it from “show top topics”), and I’ll explain why it’s prioritized."

    # Default help
    return (
        "I can help with:\n"
        "- “show units”\n"
        "- “show top topics”\n"
        "- “make a 7 day plan”\n"
        "- “did you use past papers?”\n"
        "- “what topics were extracted?”\n"
        "- “why is <topic> important?”"
    )