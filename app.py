import streamlit as st
import pandas as pd

from analyzer import analyze, read_pdf_text
from planner import top_topics, make_plan
from chatbot import answer_user
from utils import stable_hash, cache_path, load_json, save_json
from notes import has_openai_key, retrieve_relevant_chunks, generate_notes_openai

st.set_page_config(page_title="RouteIQ", page_icon="📚", layout="wide")

st.title("📚 RouteIQ — AI Study Planner Chatbot")
st.caption("Upload syllabus → select units → optional past papers → generate plan → upload materials → generate notes → ask chatbot.")

# --------------------
# Settings (simple)
# --------------------
st.sidebar.header("Study Plan")
days = st.sidebar.slider("Number of days", 3, 21, 7)

hours_per_day = st.sidebar.number_input(
    "Hours per day (optional)", min_value=0.0, max_value=16.0, value=0.0, step=0.5
)
# If hours not provided, default to 3 topic blocks/day
slots_per_day = 3 if hours_per_day == 0 else max(1, int(round(hours_per_day / 1.5)))

# --------------------
# Tabs
# --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Syllabus", "Analysis", "Study Plan", "Materials & Notes", "Chatbot"])

# Shared state
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "materials_text" not in st.session_state:
    st.session_state.materials_text = ""


def pdf_to_text(uploaded_file) -> str:
    return read_pdf_text(uploaded_file)


# --------------------
# TAB 1: SYLLABUS
# --------------------
with tab1:
    st.subheader("1) Upload Syllabus (Required)")

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

    st.subheader("2) Upload Past Papers (Optional)")
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
                "topic_frequency": res.topic_frequency,
                "selected_units": selected_units,
                "has_past_papers": bool(paper_texts),
            }
            save_json(cache_file, st.session_state.analysis)
        st.success("Analysis complete (cached).")

    # auto-load from cache if exists
    if st.session_state.analysis is None:
        cached = load_json(cache_file)
        if cached:
            st.session_state.analysis = cached


# --------------------
# TAB 2: ANALYSIS
# --------------------
with tab2:
    st.subheader("Analysis Summary")
    analysis = st.session_state.analysis

    if not analysis:
        st.info("Upload syllabus and click Run Analysis first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Units included", len(analysis["topics_by_unit"]))
        c2.metric("Past papers used", "Yes" if analysis.get("has_past_papers") else "No")
        c3.metric("Questions parsed", analysis.get("questions_count", 0))

        st.write("**Units selected:**", analysis.get("selected_units") or "All units")

        st.subheader("Top Topics")
        tt = top_topics(analysis["topic_frequency"], k=15)
        df = pd.DataFrame(tt, columns=["Topic", "Weight/Frequency"])
        st.dataframe(df, use_container_width=True)


# --------------------
# TAB 3: STUDY PLAN
# --------------------
with tab3:
    st.subheader("Study Plan")
    analysis = st.session_state.analysis

    if not analysis:
        st.info("Run analysis first.")
    else:
        st.write(f"Plan length: **{days} days**")
        st.write(f"Daily load: **{slots_per_day} topic blocks/day**" + ("" if hours_per_day == 0 else f" (~{hours_per_day} hrs/day)"))

        plan = make_plan(analysis["topic_frequency"], days=days, slots_per_day=slots_per_day)
        for pd_ in plan:
            st.write(f"**{pd_.day}:** " + (" | ".join(pd_.topics) if pd_.topics else "_No topics available_"))


# --------------------
# TAB 4: MATERIALS & NOTES
# --------------------
with tab4:
    st.subheader("Upload Study Materials (Optional)")
    st.caption("Upload class notes / textbook PDFs / PPT PDFs. Then generate notes for full unit or specific topics.")

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

                mode_map = {
                    "Full Unit Notes": "full_unit",
                    "Specific Topic Notes": "topic",
                    "Short Exam Notes": "short_exam",
                    "Definitions Only": "definitions",
                }
                mode = mode_map[note_mode]

                if not has_openai_key():
                    st.error("Notes generation needs OPENAI_API_KEY set (we keep this hidden from sidebar).")
                    st.info("You can still use the planner + chatbot without OpenAI.")
                else:
                    with st.spinner("Generating notes…"):
                        notes = generate_notes_openai(chunks, mode=mode, unit_or_topic=target, length=length)
                    if notes:
                        st.success("Notes generated.")
                        st.markdown(notes)
                    else:
                        st.error("Failed to generate notes.")


# --------------------
# TAB 5: CHATBOT
# --------------------
with tab5:
    st.subheader("Chatbot")
    analysis = st.session_state.analysis

    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "assistant", "content": "Hi! Run analysis first. Then ask: 'show top topics' or 'make a plan' or 'what units were extracted?'."}
        ]

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("Ask: 'show top topics' / 'make a 7 day plan' / 'what units?'")

    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        bot = answer_user(user_msg, analysis, days=days, slots_per_day=slots_per_day)

        st.session_state.chat.append({"role": "assistant", "content": bot})
        with st.chat_message("assistant"):
            st.write(bot)