import json
import os
from pathlib import Path

import streamlit as st

from app import generate_question, initialize_vertex, process_question_with_scenarios

PROJECT_ROOT = Path(__file__).resolve().parent
CREDENTIALS_FILE = PROJECT_ROOT / "turito-questions-4801fdc9d428.json"


def ensure_project_cwd() -> None:
    os.chdir(PROJECT_ROOT)


def resolve_image_path(output_dir: str, filename: str) -> Path | None:
    if not output_dir or not filename:
        return None
    base = Path(output_dir)
    if not base.is_absolute():
        base = PROJECT_ROOT / output_dir
    candidate = base / filename
    if candidate.is_file():
        return candidate
    return None


@st.cache_resource(show_spinner="Loading Vertex AI models…")
def load_models():
    ensure_project_cwd()
    if not CREDENTIALS_FILE.is_file():
        raise FileNotFoundError(
            f"Service account JSON not found: {CREDENTIALS_FILE.name}. "
            "Place it in the app root or set GOOGLE_APPLICATION_CREDENTIALS."
        )
    return initialize_vertex()


def run_generation(
    subject: str,
    grade: str,
    chapter: str,
    topics: str,
    number: int,
) -> dict:
    ensure_project_cwd()
    gemini_model, imagen_model = load_models()

    output_dir_name = f"{subject}_{grade}_{chapter}_nano_banana_v2"
    output_dir = str(PROJECT_ROOT / output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    gemini_response = generate_question(
        gemini_model, subject, grade, chapter, topics, number, output_dir
    )
    questions = gemini_response.get("questions", [])

    processed_questions = []
    for question in questions:
        processed = process_question_with_scenarios(
            question,
            imagen_model,
            output_dir,
            subject,
            scenario="composite",
        )
        processed_questions.append(processed)

    return {
        "status": "success",
        "message": "Questions generated successfully",
        "output_dir": output_dir_name,
        "output_dir_abs": output_dir,
        "question_count": len(processed_questions),
        "questions": processed_questions,
    }


def render_options(question: dict) -> None:
    options = question.get("options") or {}
    option_type = question.get("option_type", "text")
    if option_type == "image":
        st.caption("Options are image-based (see composite card).")
    for letter in ("A", "B", "C", "D"):
        text = options.get(letter, "")
        if text and str(text).lower() not in ("n/a", "null", "none"):
            st.markdown(f"**{letter})** {text}")


st.set_page_config(
    page_title="Image Question Generator",
    page_icon="📝",
    layout="wide",
)

st.title("Image question generator")
st.caption("Generates questions and composite cards on this host (no separate API).")

with st.sidebar:
    st.header("Deploy")
    st.markdown(
        "Run with `streamlit run main.py`. Requires `app.py`, Vertex credentials, "
        "and dependencies on the same machine. Outputs are written under the project root."
    )
    cred_ok = CREDENTIALS_FILE.is_file() or bool(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if cred_ok:
        st.success("Credentials file detected.")
    else:
        st.warning(
            f"Place `{CREDENTIALS_FILE.name}` in the project root "
            "or set `GOOGLE_APPLICATION_CREDENTIALS`."
        )

with st.form("generate_form"):
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", value="Biology")
        grade = st.text_input("Grade", value="5")
    with col2:
        chapter = st.text_input("Chapter", value="Human Anatomy & Health")
        number = st.number_input("Number of questions", min_value=1, max_value=10, value=2)
    topics = st.text_input("Topics", value="skeletal system, digestive system")
    submitted = st.form_submit_button("Generate questions", type="primary", use_container_width=True)

if submitted:
    if not all([subject.strip(), grade.strip(), chapter.strip(), topics.strip()]):
        st.error("Please fill in subject, grade, chapter, and topics.")
    elif not cred_ok:
        st.error("Missing Google Cloud credentials. See the sidebar.")
    else:
        payload = {
            "subject": subject.strip(),
            "grade": str(grade).strip(),
            "chapter": chapter.strip(),
            "topics": topics.strip(),
            "number": int(number),
        }

        with st.spinner(
            f"Generating {payload['number']} question(s) — this may take several minutes…"
        ):
            try:
                result = run_generation(
                    payload["subject"],
                    payload["grade"],
                    payload["chapter"],
                    payload["topics"],
                    payload["number"],
                )
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                st.stop()

        output_dir = result.get("output_dir", "")
        questions = result.get("questions", [])

        if not questions:
            st.warning("No questions were returned. Check logs or `question_output.json` in the output folder.")
            st.stop()

        st.success(
            f"Generated **{result.get('question_count', len(questions))}** question(s). "
            f"Output folder: `{output_dir}`"
        )

        for q in questions:
            qnum = q.get("question_num", "?")
            with st.container(border=True):
                st.subheader(f"Question {qnum}")
                st.markdown(f"**{q.get('question_text', '')}**")

                meta_cols = st.columns(3)
                meta_cols[0].caption(f"Difficulty: {q.get('difficulty', '—')}")
                meta_cols[1].caption(f"Option type: {q.get('option_type', '—')}")
                meta_cols[2].caption(f"Concepts: {q.get('concepts', '—')}")

                img_info = q.get("image_files") or {}
                filename = img_info.get("composite_card") or f"Q{qnum}_complete_card.png"
                image_path = resolve_image_path(
                    result.get("output_dir_abs", output_dir), filename
                )

                img_col, detail_col = st.columns([1, 1])
                with img_col:
                    if image_path:
                        st.image(
                            str(image_path),
                            caption="Composite question card",
                            use_container_width=True,
                        )
                    else:
                        st.info(f"Image not found: `{output_dir}/{filename}`")
                with detail_col:
                    desc = q.get("question_image_description")
                    if desc and str(desc).lower() not in ("n/a", "null", "none"):
                        st.markdown("**Question image description**")
                        st.write(desc)
                    st.markdown("**Options**")
                    render_options(q)
                    with st.expander("Answer key"):
                        st.markdown(f"**Correct:** {q.get('correct_answer', '—')}")
                        st.write(q.get("explanation", ""))

        with st.expander("Full generation result"):
            st.code(json.dumps(result, indent=2, default=str), language="json")

else:
    st.info("Fill in the form and click **Generate questions** to start.")
