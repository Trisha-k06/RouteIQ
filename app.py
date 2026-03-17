import streamlit as st
import pandas as pd

from analyzer import analyze, read_pdf_text
from planner import top_topics, make_plan
from chatbot import answer_user
from utils import stable_hash, cache_path, load_json, save_json
from notes import has_openai_key, retrieve_relevant_chunks, generate_notes_openai
from expert_system import recommend, to_dict

st.set_page_config(page_title="RouteIQ", page_icon="📚", layout="wide")


PASTEL_CSS = """
<style>
  :root{
    --bg: #fbfbff;
    --card: #ffffff;
    --stroke: #e6e8f2;
    --text: #1f2430;
    --muted: #5a6270;
    --accent: #6b7cff;      /* pastel indigo */
    --accent2: #35b2a8;     /* pastel teal */
    --warn: #d9822b;        /* accessible amber */
  }
  .stApp{
    background: radial-gradient(1200px 600px at 20% 0%, #eef0ff 0%, transparent 55%),
                radial-gradient(900px 500px at 90% 10%, #e8fbf8 0%, transparent 52%),
                var(--bg);
    color: var(--text);
  }
  .routeiq-card{
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 16px;
    padding: 16px 16px;
    box-shadow: 0 6px 18px rgba(31,36,48,0.06);
  }
  .routeiq-muted{ color: var(--muted); }
  .routeiq-pill{
    display:inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: rgba(107,124,255,0.12);
    border: 1px solid rgba(107,124,255,0.25);
    color: var(--text);
    font-size: 12px;
    margin-right: 6px;
  }
  .routeiq-footer{
    margin-top: 24px;
    padding: 14px 0 6px 0;
    text-align: center;
    color: var(--muted);
    font-size: 13px;
  }
  .routeiq-title{
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
  }
  .routeiq-subtitle{
    margin-top: 6px;
    color: var(--muted);
  }
</style>
"""

st.markdown(PASTEL_CSS, unsafe_allow_html=True)

st.markdown('<div class="routeiq-card">', unsafe_allow_html=True)
st.markdown('<div class="routeiq-title">RouteIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="routeiq-subtitle">AI Study Planner Chatbot • Knowledge Representation • Rule-based Reasoning • Planning • Expert System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="routeiq-muted">Workflow: syllabus → select units → optional past papers → analyze priorities → generate plan → optional materials → optional notes → chatbot.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------
# Tabs
# --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Syllabus", "Analysis", "Study Plan", "Materials & Notes", "Chatbot"])

# Shared state
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "materials_text" not in st.session_state:
    st.session_state.materials_text = ""
if "expert" not in st.session_state:
    st.session_state.expert = None


def pdf_to_text(uploaded_file) -> str:
    return read_pdf_text(uploaded_file)


# --------------------
# TAB 1: SYLLABUS
# --------------------
with tab1:
    st.markdown('<div class="routeiq-card">', unsafe_allow_html=True)
    st.subheader("1) Syllabus input")
    st.caption("Paste text (best) or upload a syllabus PDF. Then choose units (optional).")

    col1, col2 = st.columns(2)
    with col1:
        syllabus_text = st.text_area("Paste syllabus text here (recommended)", height=240)
    with col2:
        syllabus_pdf = st.file_uploader("Or upload syllabus PDF", type=["pdf"])

    final_syllabus_text = syllabus_text.strip()
    if (not final_syllabus_text) and syllabus_pdf is not None:
        try:
            final_syllabus_text = pdf_to_text(syllabus_pdf)
        except Exception as e:
            st.error(f"Could not read syllabus PDF: {e}")

    st.subheader("2) Past papers (optional)")
    past_papers = st.file_uploader(
        "Upload past paper PDFs (optional). If not provided, the app still generates plan using syllabus topics.",
        type=["pdf"],
        accept_multiple_files=True,
    )

    paper_texts = []
    if past_papers:
        for f in past_papers:
            try:
                paper_texts.append(pdf_to_text(f))
            except Exception as e:
                st.warning(f"Could not read {f.name}: {e}")

    # Unit selection UI (after we can parse syllabus)
    selected_units = []
    if final_syllabus_text:
        # quick parse to show units; using analyzer's extraction via analyze with no papers
        preview = analyze(final_syllabus_text, [], selected_units=None)
        units = list(preview.topics_by_unit.keys())
        if units:
            selected_units = st.multiselect(
                "Select units to focus (optional). If you don’t select anything, all units will be included.",
                options=units,
            )
        else:
            st.info("Couldn’t detect units. Paste syllabus text for best results.")

    st.divider()
    st.subheader("3) Study constraints (for planning + expert rules)")
    c1, c2 = st.columns([1, 1])
    with c1:
        days = st.slider("Number of days", 1, 21, 7)
    with c2:
        hours_per_day = st.number_input(
            "Hours per day (optional)", min_value=0.0, max_value=16.0, value=0.0, step=0.5
        )
    hours_value = None if hours_per_day == 0 else float(hours_per_day)

    st.divider()
    run = st.button("Run Analysis", type="primary", disabled=not bool(final_syllabus_text))

    # caching
    cache_key = stable_hash((final_syllabus_text or "") + "||" + "||".join(paper_texts) + "||" + "||".join(selected_units))
    cache_file = cache_path(cache_key)

    if run:
        with st.spinner("Analyzing…"):
            res = analyze(final_syllabus_text, paper_texts, selected_units=selected_units)
            st.session_state.analysis = {
                "topics_by_unit": res.topics_by_unit,
                "questions_count": len(res.questions),
                "topic_importance": res.topic_importance,
                "selected_units": selected_units,
                "has_past_papers": bool(paper_texts),
                "days": int(days),
                "hours_per_day": hours_value,
            }
            st.session_state.expert = to_dict(
                recommend(days=int(days), hours_per_day=hours_value, has_past_papers=bool(paper_texts))
            )
            save_json(cache_file, st.session_state.analysis)
        st.success("Analysis complete (cached).")

    # auto-load from cache if exists
    if st.session_state.analysis is None:
        cached = load_json(cache_file)
        if cached:
            st.session_state.analysis = cached
            # Recompute expert from cached constraints if present
            cd = int(cached.get("days", 7))
            ch = cached.get("hours_per_day", None)
            st.session_state.expert = to_dict(recommend(days=cd, hours_per_day=ch, has_past_papers=bool(cached.get("has_past_papers"))))
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------
# TAB 2: ANALYSIS
# --------------------
with tab2:
    st.markdown('<div class="routeiq-card">', unsafe_allow_html=True)
    st.subheader("Analysis (Knowledge Representation + Reasoning)")
    analysis = st.session_state.analysis

    if not analysis:
        st.info("Upload syllabus and click Run Analysis first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Units included", len(analysis["topics_by_unit"]))
        c2.metric("Past papers used", "Yes" if analysis.get("has_past_papers") else "No")
        c3.metric("Questions parsed", analysis.get("questions_count", 0))

        st.write("**Units selected:**", analysis.get("selected_units") or "All units")

        st.subheader("Topic priorities (inferred importance)")
        tt = top_topics(analysis["topic_importance"], k=15)
        df = pd.DataFrame(tt, columns=["Topic", "Importance"])
        st.dataframe(df, use_container_width=True)

        st.subheader("Expert recommendation (rule-based)")
        expert = st.session_state.expert or {}
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown(f"**Strategy:** {expert.get('strategy','')}")
            st.markdown(f"**Focus:** {expert.get('focus_mode','')}")
        with ec2:
            st.markdown(f"**Suggested notes type:** {expert.get('suggested_notes_type','')}")
            st.markdown(f"**Planning advice:** {expert.get('planning_advice','')}")
        st.caption(expert.get("reasoning", ""))

        st.subheader("Structured syllabus view (Unit → Topics)")
        rows = []
        for unit, topics in analysis["topics_by_unit"].items():
            for tp in topics:
                key = f"{unit}: {tp}"
                rows.append({"Unit": unit, "Topic": tp, "Importance": float(analysis["topic_importance"].get(key, 1.0))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------
# TAB 3: STUDY PLAN
# --------------------
with tab3:
    st.markdown('<div class="routeiq-card">', unsafe_allow_html=True)
    st.subheader("Study plan (Planning)")
    analysis = st.session_state.analysis

    if not analysis:
        st.info("Run analysis first.")
    else:
        days = int(analysis.get("days", 7))
        hours_value = analysis.get("hours_per_day", None)
        plan = make_plan(analysis["topic_importance"], days=days, hours_per_day=hours_value)
        slots_per_day = plan[0].total_slots if plan else 0
        st.write(f"Plan length: **{days} days**")
        if hours_value is None:
            st.write(f"Daily load: **{slots_per_day} topic blocks/day** (default topic-block approach)")
        else:
            st.write(f"Daily load: **{slots_per_day} topic blocks/day** (~{hours_value} hrs/day)")

        for pd_ in plan:
            st.write(f"**{pd_.day}:** " + (" | ".join(pd_.topics) if pd_.topics else "_No topics available_"))
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------
# TAB 4: MATERIALS & NOTES
# --------------------
with tab4:
    st.markdown('<div class="routeiq-card">', unsafe_allow_html=True)
    st.subheader("Materials & Notes (Offline retrieval + optional OpenAI)")
    st.caption("Upload class notes / textbook PDFs. Retrieval works offline. OpenAI is used only when you click Generate Notes.")

    materials = st.file_uploader(
        "Upload materials PDFs (optional)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    material_texts = []
    if materials:
        for f in materials:
            try:
                material_texts.append(pdf_to_text(f))
            except Exception as e:
                st.warning(f"Could not read {f.name}: {e}")

    if material_texts:
        st.session_state.materials_text = "\n\n".join(material_texts)

    analysis = st.session_state.analysis
    if not analysis:
        st.info("Run syllabus analysis first so I know your units/topics.")
    else:
        all_topic_labels = []
        for u, tps in analysis["topics_by_unit"].items():
            for tp in tps:
                all_topic_labels.append(f"{u}: {tp}")

        st.divider()
        st.subheader("Generate Notes")

        note_mode = st.radio(
            "What do you want to generate?",
            ["Full Unit Notes", "Specific Topic Notes", "Short Exam Notes", "Definitions Only"],
            horizontal=True,
        )

        length = st.selectbox("Detail level", ["short", "medium", "long"], index=1)

        target = None
        if note_mode == "Full Unit Notes":
            units = list(analysis["topics_by_unit"].keys())
            target = st.selectbox("Select unit", units)
            query = f"{target} notes"
        else:
            target = st.selectbox("Select topic", all_topic_labels)
            query = target

        if st.button("Generate Notes", type="primary"):
            if not st.session_state.materials_text.strip():
                st.warning("Upload at least one material PDF to generate notes.")
            else:
                # Retrieve relevant chunks offline → then OPTIONAL OpenAI summarize
                chunks = retrieve_relevant_chunks(st.session_state.materials_text, query=query, k=6)
                with st.expander("Retrieved material excerpts (offline)", expanded=False):
                    if not chunks:
                        st.write("No relevant excerpts found.")
                    else:
                        for i, ch in enumerate(chunks, 1):
                            st.markdown(f"**Excerpt {i}**")
                            st.write(ch)

                mode_map = {
                    "Full Unit Notes": "full_unit",
                    "Specific Topic Notes": "topic",
                    "Short Exam Notes": "short_exam",
                    "Definitions Only": "definitions",
                }
                mode = mode_map[note_mode]

                if not has_openai_key():
                    st.info(
                        "OpenAI notes generation is optional. To enable it, set `OPENAI_API_KEY` in your environment. "
                        "You can still use analysis + planning + chatbot without it."
                    )
                else:
                    with st.spinner("Generating notes…"):
                        notes = generate_notes_openai(chunks, mode=mode, unit_or_topic=target, length=length)
                    if notes:
                        st.success("Notes generated.")
                        st.markdown(notes)
                    else:
                        st.error("Failed to generate notes.")
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------
# TAB 5: CHATBOT
# --------------------
with tab5:
    st.markdown('<div class="routeiq-card">', unsafe_allow_html=True)
    st.subheader("Chatbot (grounded on your analysis)")
    analysis = st.session_state.analysis

    if "chat" not in st.session_state:
        st.session_state.chat = [
            {
                "role": "assistant",
                "content": "Hi! Run analysis first. Then ask: “show units”, “what topics were extracted”, “show top topics”, “make a 7 day plan”, “did you use past papers?”, “why is <topic> important?”.",
            }
        ]

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("Ask: 'show top topics' / 'make a 7 day plan' / 'what units?'")

    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        days = int((analysis or {}).get("days", 7))
        hours_value = (analysis or {}).get("hours_per_day", None)
        bot = answer_user(user_msg, analysis, days=days, hours_per_day=hours_value)

        st.session_state.chat.append({"role": "assistant", "content": bot})
        with st.chat_message("assistant"):
            st.write(bot)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="routeiq-footer">Developed by Trisha Kumar</div>', unsafe_allow_html=True)