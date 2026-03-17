import os
import re
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def has_openai_key() -> bool:
    return OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks


def retrieve_relevant_chunks(material_text: str, query: str, k: int = 5) -> List[str]:
    chunks = chunk_text(material_text)
    if not chunks:
        return []
    corpus = chunks + [query]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[-1], X[:-1]).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return [chunks[i] for i in top_idx]


def build_notes_prompt(context_chunks: List[str], mode: str, unit_or_topic: str, length: str) -> str:
    """
    Compact prompt; avoids repeated token-heavy calls.
    """
    context = "\n\n".join(context_chunks)[:12000]
    instruction = {
        "full_unit": "Create well-structured unit notes with headings and subpoints.",
        "topic": "Create focused notes ONLY for the requested topic.",
        "short_exam": "Create exam-oriented short notes: crisp bullets, key steps, typical question angles.",
        "definitions": "List important definitions and key terms with 1–2 line explanations.",
    }[mode]
    len_hint = {"short": "Keep it short.", "medium": "Moderate detail.", "long": "Detailed notes."}[length]

    return (
        "You are a study notes generator.\n"
        f"Task: {instruction}\n"
        f"{len_hint}\n\n"
        f"Target: {unit_or_topic}\n\n"
        "Use ONLY the provided material context. If something is missing, say: Not found in material.\n"
        "Return clean markdown with headings and bullet points.\n\n"
        "MATERIAL CONTEXT:\n"
        f"{context}"
    ).strip()


def generate_notes_openai(
    context_chunks: List[str],
    mode: str,
    unit_or_topic: str,
    length: str = "medium",
) -> Optional[str]:
    """
    mode: "full_unit" | "topic" | "short_exam" | "definitions"
    length: "short" | "medium" | "long"
    """
    if not has_openai_key():
        return None

    client = OpenAI()

    prompt = build_notes_prompt(context_chunks=context_chunks, mode=mode, unit_or_topic=unit_or_topic, length=length)

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.2,
    )
    return resp.output_text.strip()