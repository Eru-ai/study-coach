"""
Study Coach Agent — full loop with feedback pause.
  PLAN -> QUIZ -> [answer] -> JUDGE -> DECIDE -> loop -> REPORT
Run with: streamlit run study_coach_app.py
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

MODEL_NAME = "gemini-2.5-flash-lite"

st.set_page_config(page_title="Study Coach", page_icon="🎓", layout="centered")
st.title("🎓 Study Coach")
st.caption("Give it a topic. It plans your revision, quizzes you, and adapts to what you're weak on.")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("No GEMINI_API_KEY found.")
    st.stop()
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)


# ========== AGENT MEMORY ==========
def init_state():
    defaults = {
        "plan": [],
        "current_index": 0,
        "question": "",
        "attempts_on_current": 0,
        "results": [],
        "finished": False,
        "pending_feedback": None,
        "subject": "",
        "topic": "",
        "level": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ========== AGENT STEPS ==========
def make_plan(subject, topic, level):
    prompt = f"""You are an experienced NCEA teacher in New Zealand planning a revision session.
Subject: {subject}
Topic: {topic}
Level: {level}

Break this topic into 4 subtopics a student should master, ordered foundational to advanced.
Respond with ONLY a JSON array of strings. No markdown, no backticks.
Example: ["Subtopic one", "Subtopic two", "Subtopic three", "Subtopic four"]"""
    raw = model.generate_content(prompt).text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def make_question(subtopic):
    prompt = f"""You are an experienced NCEA teacher in New Zealand.
Subject: {st.session_state.subject}
Overall topic: {st.session_state.topic}
Level: {st.session_state.level}
Subtopic to test: {subtopic}

Write ONE question testing whether the student truly understands this subtopic —
requiring explanation/reasoning, not a one-word answer.
Respond with JUST the question."""
    return model.generate_content(prompt).text.strip()


def judge_answer(subtopic, question, answer):
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
    lines.append(f"You worked through **{len(st.session_state.results)}** subtopics.\n")
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


def advance():
    st.session_state.current_index += 1
    st.session_state.attempts_on_current = 0
    if st.session_state.current_index >= len(st.session_state.plan):
        st.session_state.finished = True
        st.session_state.question = ""
    else:
        st.session_state.question = make_question(
            st.session_state.plan[st.session_state.current_index]
        )


# ========== UI: START ==========
if not st.session_state.plan:
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", placeholder="e.g. NCEA Level 2 Physics")
    with col2:
        topic = st.text_input("Topic", placeholder="e.g. Momentum")
    level = st.select_slider("Difficulty", ["Easy", "Medium", "Hard", "Excellence-level"], value="Medium")

    if st.button("🚀 Start session", type="primary", use_container_width=True):
        if not subject.strip() or not topic.strip():
            st.warning("Fill in both subject and topic.")
        else:
            with st.spinner("Planning your session..."):
                try:
                    st.session_state.subject = subject
                    st.session_state.topic = topic
                    st.session_state.level = level
                    st.session_state.plan = make_plan(subject, topic, level)
                    st.session_state.question = make_question(st.session_state.plan[0])
                    st.rerun()
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

# ========== UI: SESSION ==========
else:
    st.subheader("📋 Revision plan")
    for i, sub in enumerate(st.session_state.plan):
        if i < st.session_state.current_index:
            st.markdown(f"✅ **{i+1}.** {sub}")
        elif i == st.session_state.current_index and not st.session_state.finished:
            st.markdown(f"👉 **{i+1}.** {sub}")
        else:
            st.markdown(f"　 **{i+1}.** {sub}")

    st.divider()

    if st.session_state.finished:
        st.markdown(make_report())
        if st.button("🔄 New session", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    else:
        st.subheader(f"❓ Question {st.session_state.current_index + 1}")
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
                                "note": "Have another go — try to fill the gaps above.",
                                "advance": False,
                            }
                            st.rerun()