import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class AnalysisResult:
    topics_by_unit: Dict[str, List[str]]
    questions: List[str]
    question_topic: List[Tuple[str, str, float]]  # (question, mapped_topic, confidence)
    topic_frequency: Dict[str, int]


def read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(file_bytes)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)


def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def extract_units_and_topics(syllabus_text: str) -> Dict[str, List[str]]:
    """
    Heuristic parser:
    - Split by 'UNIT' occurrences
    - Extract lines that look like topic bullets or comma-separated phrases
    Works surprisingly well for most college syllabi.
    """
    text = normalize_text(syllabus_text)
    # Make "UNIT" consistent
    text = re.sub(r"\bUnit\b", "UNIT", text, flags=re.IGNORECASE)

    # Split into units
    parts = re.split(r"\bUNIT\s*[-:]?\s*(\d+)\b", text)
    # parts like: [before, unit_num, content, unit_num, content, ...]
    topics_by_unit: Dict[str, List[str]] = {}

    if len(parts) < 3:
        # fallback: treat whole thing as one unit
        topics_by_unit["Unit 1"] = extract_topic_lines(text)
        return topics_by_unit

    # iterate pairs of (unit_num, content)
    for i in range(1, len(parts), 2):
        unit_num = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        unit_key = f"Unit {unit_num}"
        topics = extract_topic_lines(content)
        topics_by_unit[unit_key] = dedupe_keep_order(topics)

    # Remove empty topic lists if any
    topics_by_unit = {u: t for u, t in topics_by_unit.items() if len(t) > 0}
    if not topics_by_unit:
        topics_by_unit["Unit 1"] = extract_topic_lines(text)
    return topics_by_unit


def extract_topic_lines(unit_text: str) -> List[str]:
    t = normalize_text(unit_text)

    lines = [normalize_text(x) for x in t.split("\n") if normalize_text(x)]
    topics: List[str] = []

    for line in lines:
        # remove common noise
        line = re.sub(r"(hours|lecture|credits?)\s*[:\-]?\s*\d+", "", line, flags=re.IGNORECASE).strip()

        # bullet-like or colon-separated
        if re.match(r"^(\-|\•|\*)\s+", line):
            line = re.sub(r"^(\-|\•|\*)\s+", "", line).strip()
            topics.extend(split_topics(line))
        elif ":" in line and len(line) < 120:
            # "Topics: A, B, C"
            right = line.split(":", 1)[1].strip()
            topics.extend(split_topics(right))
        else:
            # Sometimes syllabus lists topics as a sentence with commas
            if 10 < len(line) < 140 and ("," in line or ";" in line):
                topics.extend(split_topics(line))

    # Fallback if nothing extracted
    if not topics:
        topics = split_topics(t)

    # Clean topics
    cleaned = []
    for x in topics:
        x = re.sub(r"\s+", " ", x).strip(" -•*\t\r\n")
        x = re.sub(r"^\d+[\.\)]\s*", "", x).strip()
        if 2 <= len(x) <= 80:
            cleaned.append(x)
    return cleaned


def split_topics(s: str) -> List[str]:
    # Split by commas/semicolons but keep meaningful phrases
    parts = re.split(r"[;,]\s*", s)
    # Further split by "and" cautiously
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Avoid splitting something like "time and space complexity" into nonsense:
        if " and " in p and len(p) > 35:
            out.append(p)
        else:
            sub = [x.strip() for x in p.split(" and ") if x.strip()]
            out.extend(sub)
    return out


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def extract_questions(paper_text: str) -> List[str]:
    """
    Split into question-like chunks using common patterns:
    Q1, 1., 1), Part A, etc.
    """
    t = normalize_text(paper_text)

    # Add newlines before question markers to help splitting
    t = re.sub(r"(\bQ\s*\d+\b)", r"\n\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(\n?\b\d+\s*[\.\)]\s+)", r"\n\1", t)

    chunks = [normalize_text(c) for c in t.split("\n") if normalize_text(c)]
    questions = []
    buf = ""

    def flush():
        nonlocal buf
        if len(buf) >= 20:
            questions.append(buf.strip())
        buf = ""

    for c in chunks:
        is_new_q = bool(re.match(r"^(Q\s*\d+|\d+\s*[\.\)])\s*", c, flags=re.IGNORECASE))
        if is_new_q:
            flush()
            buf = c
        else:
            buf += " " + c

    flush()

    # Sometimes papers don’t have clean markers; fallback split by '?'
    if len(questions) < 3:
        questions = [q.strip() for q in re.split(r"\?\s*", t) if len(q.strip()) > 30]
        questions = [q + "?" for q in questions[:60]]

    return questions[:120]  # cap for speed


def map_questions_to_topics(
    questions: List[str],
    topics_by_unit: Dict[str, List[str]],
    threshold: float = 0.16,
) -> Tuple[List[Tuple[str, str, float]], Dict[str, int]]:
    """
    TF-IDF similarity: question vs topic strings.
    Returns mapping with confidence.
    """
    all_topics = []
    for unit, topics in topics_by_unit.items():
        for tp in topics:
            all_topics.append(f"{unit}: {tp}")

    if not questions or not all_topics:
        return [], {}

    # Fit vectorizer on both topics and questions
    corpus = all_topics + questions
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(corpus)

    topic_vecs = X[: len(all_topics)]
    q_vecs = X[len(all_topics) :]

    sims = cosine_similarity(q_vecs, topic_vecs)  # (num_q, num_topics)

    mapped = []
    freq = {t: 0 for t in all_topics}

    for i, q in enumerate(questions):
        j = int(np.argmax(sims[i]))
        conf = float(sims[i, j])
        topic = all_topics[j] if conf >= threshold else "UNMAPPED"
        mapped.append((q, topic, conf))
        if topic != "UNMAPPED":
            freq[topic] += 1

    # remove zero freq for neatness
    freq = {k: v for k, v in freq.items() if v > 0}
    return mapped, freq


def analyze(
    syllabus_text: str,
    paper_texts: List[str],
    selected_units: List[str] = None,
) -> AnalysisResult:
    topics_by_unit = extract_units_and_topics(syllabus_text)

    if selected_units:
        topics_by_unit = filter_units(topics_by_unit, selected_units)

    all_questions: List[str] = []
    for pt in (paper_texts or []):
        all_questions.extend(extract_questions(pt))

    all_questions = dedupe_keep_order([q[:400] for q in all_questions])

    if all_questions:
        q_topic, freq = map_questions_to_topics(all_questions, topics_by_unit)
    else:
        q_topic = []
        freq = syllabus_only_frequency(topics_by_unit)

    return AnalysisResult(
        topics_by_unit=topics_by_unit,
        questions=all_questions,
        question_topic=q_topic,
        topic_frequency=freq,
    )
def filter_units(topics_by_unit, selected_units):
    if not selected_units:
        return topics_by_unit
    return {u: topics_by_unit[u] for u in selected_units if u in topics_by_unit}


def syllabus_only_frequency(topics_by_unit):
    """
    If no past papers, we still need 'importance' weights.
    Simple MVP: give each topic weight 1.
    (Later you can improve: weight by subtopic length or keywords.)
    """
    freq = {}
    for unit, topics in topics_by_unit.items():
        for tp in topics:
            freq[f"{unit}: {tp}"] = 1
    return freq