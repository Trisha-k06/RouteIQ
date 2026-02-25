import os
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def is_openai_ready() -> bool:
    return OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))


def ai_extract_topics(syllabus_text: str) -> Optional[Dict[str, List[str]]]:
    """
    One call to cleanly extract topics if heuristic parser fails.
    """
    if not is_openai_ready():
        return None

    client = OpenAI()
    prompt = (
        "Extract a unit-wise topic list from this syllabus text.\n"
        "Return STRICT JSON only in this format:\n"
        '{ "Unit 1": ["Topic A", "Topic B"], "Unit 2": ["Topic C"] }\n\n'
        f"Syllabus:\n{syllabus_text[:6000]}"
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.1,
    )

    text = resp.output_text.strip()
    # Expect JSON text
    import json
    try:
        return json.loads(text)
    except Exception:
        return None