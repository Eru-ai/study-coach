"""
Study Coach Agent — full loop with feedback pause.
  PLAN -> QUIZ -> [answer] -> JUDGE -> DECIDE -> loop -> REPORT
v9: plain-text maths in questions (no LaTeX) + robust past-paper reader + target grade + level badge
Run with: streamlit run study_coach_app.py
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from docx import Document

MODEL_NAME = "gemini-2.5-flash-lite"

st.set_page_config(page_title="Study Coach", page_icon="🎓", layout="centered")
st.title("🎓 Study Coach")
st.caption("Give it a topic. It plans your revision, quizzes you, and adapts to what you're weak on.")

with st.expander("ℹ️ How to use"):
    st.markdown(
        """
**What this does:** Type any subject and topic. Study Coach builds a short revision
plan, then quizzes you one subtopic at a time — and shows what you've nailed and what
needs more work.

**Steps:**
1. Enter a **subject** (e.g. NCEA Level 2 Physics) and a **topic** (e.g. Momentum).
2. Pick the **grade you're aiming for** (Achieved, Merit, or Excellence).
3. *(Optional)* Paste a **past paper** — questions will match its style and level.
4. Tick **Answer-only mode** if it's maths — work on paper, type just your final answer.
5. Hit **Start session**. Each question shows the level it's testing you at.
6. You get **2 tries** per question, then a summary of your strong and weak areas.

**Tip:** Be honest with your answers — it can only find your weak spots if you really try.

**Remember:** AI can make mistakes. Use this to guide your revision, not as the final
word — check anything important with your teacher.
        """
    )

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("No GEMINI_API_KEY found.")
    st.stop()
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)


# ========== FILE READING ==========
def read_paper_file(uploaded_file):
    """Pull text out of an uploaded past paper.
    Detects the REAL format from the file's contents, not its name, so it
    handles genuine PDFs, Word docs, plain text, and zip-based scan bundles
    (some scanner apps export a .pdf that is really a zip of page images + text)."""
    import io, zipfile
    data = uploaded_file.getvalue()

    # Genuine PDF (starts with %PDF-)
    if data[:5] == b"%PDF-":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    # ZIP-based file (starts with PK) — could be a real .docx or a scan bundle
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            if "[Content_Types].xml" in names:
                # Real Word document
                doc = Document(io.BytesIO(data))
                return "\n".join(p.text for p in doc.paragraphs)
            # Scan bundle: gather the per-page text files inside
            txts = sorted(n for n in names if n.lower().endswith(".txt"))
            if txts:
                return "\n".join(z.read(n).decode("utf-8", errors="ignore") for n in txts)
            return ""  # only images inside, no readable text

    # Plain text fallback
    return data.decode("utf-8", errors="ignore")


# ========== AGENT MEMORY ==========
def init_state():
    defaults = {
        "plan": [],
        "current_index": 0,
        "question": "",
        "question_level": "",
        "attempts_on_current": 0,
        "results": [],
        "finished": False,
        "pending_feedback": None,
        "subject": "",
        "topic": "",
        "target_grade": "Merit",
        "past_paper": "",
        "answer_only": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ========== AGENT STEPS ==========
def make_plan(subject, topic, target_grade):
    paper = st.session_state.past_paper.strip()
    paper_block = (
        "\nA past paper / practice test was provided below. Base the subtopics on what it "
        "actually covers, so revision matches the real assessment:\n---\n"
        f"{paper}\n---\n"
        if paper else ""
    )
    prompt = f"""You are an experienced NCEA teacher in New Zealand planning a revision session.
Subject: {subject}
Topic: {topic}
Grade the student is aiming for: {target_grade}
{paper_block}
Break this topic into 4 subtopics a student should master, ordered foundational to advanced.
Respond with ONLY a JSON array of strings. No markdown, no backticks.
Example: ["Subtopic one", "Subtopic two", "Subtopic three", "Subtopic four"]"""
    raw = model.generate_content(prompt).text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def make_question(subtopic):
    paper = st.session_state.past_paper.strip()
    paper_block = (
        "\nUSE THIS PAST PAPER as your reference for style, format, and difficulty. "
        "Write a question that closely matches how questions are asked in it:\n---\n"
        f"{paper}\n---\n"
        if paper else ""
    )
    if st.session_state.answer_only:
        style = (
            "The question must have a clear, specific FINAL ANSWER "
            "(a number, an expression, or a short result). "
            "The student works it out on paper and types ONLY their final answer, "
            "so it must be solvable to one definite answer. Do not ask them to explain reasoning."
        )
    else:
        style = (
            "The question must test whether the student truly understands this subtopic — "
            "requiring explanation/reasoning, not a one-word answer."
        )
    prompt = f"""You are an experienced NCEA teacher in New Zealand.
Subject: {st.session_state.subject}
Overall topic: {st.session_state.topic}
Grade the student is aiming for: {st.session_state.target_grade}
Subtopic to test: {subtopic}
{paper_block}
Write ONE question aimed at the {st.session_state.target_grade} level. {style}

FORMATTING: Write ALL maths in plain text — NO LaTeX and no backslash commands.
Use "/" for fractions and "^" for powers, plus Unicode where helpful (superscripts, sqrt, pi, times).
For example write  (2x-1)/(4x^2-1)  as plain text, never as a LaTeX frac command.

Respond with ONLY JSON, no markdown, no backticks:
{{"question": "the full question text", "level": "Achieved" or "Merit" or "Excellence"}}
"level" is the NCEA grade level this question targets (usually the grade they're aiming for)."""
    raw = model.generate_content(prompt).text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    return data["question"], data.get("level", st.session_state.target_grade)


def judge_answer(subtopic, question, answer):
    if st.session_state.answer_only:
        prompt = f"""You are an experienced NCEA teacher marking a student's FINAL ANSWER.
Subtopic: {subtopic}
Question: {question}
Student's final answer: {answer}

The student worked the problem out on paper and typed only their final answer.
Judge whether their FINAL ANSWER is correct. Accept equivalent forms and sensible rounding.
Respond with ONLY JSON, no markdown:
{{
  "verdict": "solid" or "shaky",
  "feedback": "2-3 sentences. If correct, confirm it. If wrong, give the correct answer and one hint about the likely mistake. Plain text maths only, no LaTeX."
}}
Use "solid" only if the final answer is correct."""
    else:
        prompt = f"""You are an experienced NCEA teacher marking a student's answer.
Subtopic: {subtopic}
Question: {question}
Student's answer: {answer}

Assess whether they understand this well enough to move on.
Respond with ONLY JSON, no markdown:
{{
  "verdict": "solid" or "shaky",
  "feedback": "2-3 sentences: what's right, what's missing. Plain text maths only, no LaTeX."
}}
Use "solid" only if the answer shows real understanding."""
    raw = model.generate_content(prompt).text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def make_report():
    lines = ["## 📊 Session summary\n"]
    solid = [r for r in st.session_state.results if r["verdict"] == "solid"]
    shaky = [r for r in st.session_state.results if r["verdict"] == "shaky"]
    lines.append(f"You worked through **{len(st.session_state.results)}** subtopics "
                 f"aiming for **{st.session_state.target_grade}**.\n")
    if solid:
        lines.append("**✅ Solid:**")
        for r in solid:
            lines.append(f"- {r['subtopic']}")
    if shaky:
        lines.append("\n**⚠️ Needs more work:**")
        for r in shaky:
            lines.append(f"- {r['subtopic']}")
    lines.append("\nFocus your revision on the 'needs more work' areas.")
    return "\n".join(lines)


def level_badge(level):
    color = {"Achieved": "blue", "Merit": "violet", "Excellence": "green"}.get(level, "grey")
    return f":{color}-background[**{level} level**]"


def render_journey():
    """Draw the revision plan as a progress journey."""
    total = len(st.session_state.plan)
    if st.session_state.finished:
        st.progress(1.0, text="Complete! 🎉")
    else:
        step_now = st.session_state.current_index + 1
        st.progress(st.session_state.current_index / total, text=f"Step {step_now} of {total}")

    for i, sub in enumerate(st.session_state.plan):
        if i < st.session_state.current_index:
            st.markdown(f":green[✅ **{i+1}.** {sub}]")
        elif i == st.session_state.current_index and not st.session_state.finished:
            st.markdown(f"🎯 **{i+1}. {sub}** &nbsp;:blue[← you're here]")
        else:
            st.markdown(f":grey[⚪ **{i+1}.** {sub}]")


def advance():
    st.session_state.current_index += 1
    st.session_state.attempts_on_current = 0
    if st.session_state.current_index >= len(st.session_state.plan):
        st.session_state.finished = True
        st.session_state.question = ""
        st.session_state.question_level = ""
    else:
        q, lvl = make_question(st.session_state.plan[st.session_state.current_index])
        st.session_state.question = q
        st.session_state.question_level = lvl


# ========== UI: START ==========
if not st.session_state.plan:
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", placeholder="e.g. NCEA Level 2 Physics")
    with col2:
        topic = st.text_input("Topic", placeholder="e.g. Momentum")
    target_grade = st.select_slider("Grade you're aiming for", ["Achieved", "Merit", "Excellence"], value="Merit")

    with st.expander("📄 Add a past paper or practice test (optional)"):
        st.caption("Upload a past paper (PDF, Word, or text) — or paste it below. "
                   "The coach will make questions in the same style and at your target level.")
        paper_file = st.file_uploader(
            "Upload a past paper",
            type=["pdf", "docx", "txt"],
            key="paper_upload",
        )
        pasted_paper = st.text_area(
            "Or paste past paper text",
            height=140,
            placeholder="Paste past exam questions here...",
        )
        # Uploaded file wins if both are given
        if paper_file is not None:
            try:
                past_paper = read_paper_file(paper_file)
                if past_paper.strip():
                    st.success(f"Loaded {paper_file.name} ({len(past_paper)} characters).")
                else:
                    st.warning("Could not read any text from that file — it might be a scanned image. Try pasting instead.")
            except Exception as e:
                past_paper = ""
                st.error(f"Could not read that file: {e}")
        else:
            past_paper = pasted_paper

    answer_only = st.checkbox(
        "⚡ Answer-only mode — best for maths. Work it out on paper, type just the final answer."
    )

    if st.button("🚀 Start session", type="primary", use_container_width=True):
        if not subject.strip() or not topic.strip():
            st.warning("Fill in both subject and topic.")
        else:
            with st.spinner("Planning your session..."):
                try:
                    st.session_state.subject = subject
                    st.session_state.topic = topic
                    st.session_state.target_grade = target_grade
                    st.session_state.past_paper = past_paper
                    st.session_state.answer_only = answer_only
                    st.session_state.plan = make_plan(subject, topic, target_grade)
                    q, lvl = make_question(st.session_state.plan[0])
                    st.session_state.question = q
                    st.session_state.question_level = lvl
                    st.rerun()
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

# ========== UI: SESSION ==========
else:
    st.subheader("📋 Your revision journey")
    render_journey()

    st.divider()

    if st.session_state.finished:
        st.markdown(make_report())
        if st.button("🔄 New session", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    else:
        st.subheader(f"❓ Question {st.session_state.current_index + 1}")
        if st.session_state.question_level:
            st.markdown(level_badge(st.session_state.question_level))
        st.info(st.session_state.question)

        fb = st.session_state.pending_feedback
        if fb:
            # Show frozen feedback + a button to continue
            if fb["verdict"] == "solid":
                st.success(fb["text"])
            else:
                st.warning(fb["text"])
                if fb.get("note"):
                    st.info(fb["note"])

            label = "Next question →" if fb["advance"] else "Try again →"
            if st.button(label, type="primary", use_container_width=True):
                st.session_state.pending_feedback = None
                if fb["advance"]:
                    advance()
                st.rerun()

        else:
            if st.session_state.answer_only:
                answer = st.text_input(
                    "Your final answer (work it out on paper, then type the answer)",
                    key=f"ans_{st.session_state.current_index}_{st.session_state.attempts_on_current}",
                )
            else:
                answer = st.text_area(
                    "Your answer",
                    height=150,
                    key=f"ans_{st.session_state.current_index}_{st.session_state.attempts_on_current}",
                )
            if st.button("Submit answer", type="primary", use_container_width=True):
                if not answer.strip():
                    st.warning("Write an answer first.")
                else:
                    with st.spinner("Marking..."):
                        result = judge_answer(
                            st.session_state.plan[st.session_state.current_index],
                            st.session_state.question,
                            answer,
                        )
                    st.session_state.attempts_on_current += 1

                    if result["verdict"] == "solid":
                        st.session_state.results.append({
                            "subtopic": st.session_state.plan[st.session_state.current_index],
                            "verdict": "solid",
                            "attempts": st.session_state.attempts_on_current,
                        })
                        st.session_state.pending_feedback = {
                            "verdict": "solid", "text": result["feedback"], "advance": True,
                        }
                        st.rerun()
                    else:
                        if st.session_state.attempts_on_current >= 2:
                            st.session_state.results.append({
                                "subtopic": st.session_state.plan[st.session_state.current_index],
                                "verdict": "shaky",
                                "attempts": st.session_state.attempts_on_current,
                            })
                            st.session_state.pending_feedback = {
                                "verdict": "shaky", "text": result["feedback"],
                                "note": "Noted this as one to revisit. Let's move on.",
                                "advance": True,
                            }
                            st.rerun()
                        else:
                            st.session_state.pending_feedback = {
                                "verdict": "shaky", "text": result["feedback"],
                                "note": "Have another go — check your working and try again.",
                                "advance": False,
                            }
                            st.rerun()