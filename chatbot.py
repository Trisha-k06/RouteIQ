from typing import Any, Dict, Optional
from planner import top_topics, make_plan


def answer_user(
    message: str,
    analysis: Optional[Dict[str, Any]],
    days: int,
    slots_per_day: int,
) -> str:
    msg = message.lower().strip()

    if analysis is None:
        return "Upload syllabus and click Run Analysis first. Past papers are optional."

    freq = analysis.get("topic_frequency", {})
    topics_by_unit = analysis.get("topics_by_unit", {})

    # 1) Units listing
    if ("unit" in msg or "units" in msg) and ("show" in msg or "list" in msg):
        units = list(topics_by_unit.keys())
        if not units:
            return "I couldn't detect any units from the syllabus. Try pasting syllabus text instead of PDF."
        return "Units detected:\n" + "\n".join([f"- {u}" for u in units])

    # 2) Past papers status
    if "past paper" in msg or "past papers" in msg:
        if analysis.get("has_past_papers"):
            return "Yes — I used past papers to rank topics by frequency."
        return "No past papers were uploaded. I’m ranking topics using syllabus topics only."

    # 3) Top topics / high-yield
    if "top" in msg and ("topic" in msg or "high yield" in msg or "important" in msg):
        tt = top_topics(freq, k=10)
        if not tt:
            return "I couldn't rank topics yet. Please run analysis again (and optionally upload past papers)."
        lines = ["Here are the top high-yield topics:"]
        for i, (t, f) in enumerate(tt, 1):
            lines.append(f"{i}. {t}  (score {f})")
        return "\n".join(lines)

    # 4) Plan request
    if "plan" in msg or "schedule" in msg or "timetable" in msg:
        plan = make_plan(freq, days=days, slots_per_day=slots_per_day)
        lines = [f"Here’s a {days}-day plan ({slots_per_day} topic blocks/day):"]
        for pd in plan:
            lines.append(f"- {pd.day}: " + (" | ".join(pd.topics) if pd.topics else "(no topics found)"))
        return "\n".join(lines)

    # 5) Syllabus summary
    if "syllabus" in msg or "topics extracted" in msg or ("what" in msg and "topics" in msg):
        lines = ["Here’s what I extracted from your syllabus:"]
        for unit, topics in topics_by_unit.items():
            lines.append(f"- {unit}: {len(topics)} topics")
        lines.append("Try: “show units”, “show top topics”, “make a 7 day plan”.")
        return "\n".join(lines)

    # 6) Why important?
    if "why" in msg and ("important" in msg or "priority" in msg):
        tt = top_topics(freq, k=25)
        for t, f in tt:
            topic_name = t.split(":", 1)[-1].strip().lower()
            if topic_name and topic_name in msg:
                if analysis.get("has_past_papers"):
                    return f"'{t}' is ranked high because it appears {f} times across uploaded past papers."
                return f"'{t}' is included as a priority topic from your selected syllabus units (score {f})."
        return "Copy-paste the exact topic name from “show top topics”, and I’ll explain why it’s prioritized."

    # Default help
    return (
        "I can help with:\n"
        "- “show units”\n"
        "- “show top topics”\n"
        "- “make a 7 day plan”\n"
        "- “did you use past papers?”\n"
        "- “why is <topic> important?”"
    )