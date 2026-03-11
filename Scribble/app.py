from __future__ import annotations

import os
import hashlib

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from services.ai_service import AIService, FallbackProcessingError, QuotaExceededError, get_api_key_from_env
from services.export_service import (
    build_csv_bytes,
    build_docx_bytes,
    build_pdf_bytes,
    build_txt_bytes,
)
from services.ocr_service import OCRService


load_dotenv()

st.set_page_config(page_title="Scribble to Digital", page_icon="📝", layout="wide")


def _init_state() -> None:
    defaults = {
        "ocr_text": "",
        "clean_notes": "",
        "tasks": [],
        "model": "",
        "image_name": "",
        "processed_preview": None,
        "raw_response": "",
        "enhancement_mode": "balanced",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _read_image(uploaded_file) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(uploaded_file).convert("RGB")
    return image, np.array(image)





def _render_downloads(clean_notes: str, tasks: list[str], image_name: str) -> None:
    stem = os.path.splitext(image_name or "scribble_output")[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.download_button(
        "Download TXT",
        data=build_txt_bytes(clean_notes, tasks),
        file_name=f"{stem}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    col2.download_button(
        "Download CSV",
        data=build_csv_bytes(tasks),
        file_name=f"{stem}_tasks.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col3.download_button(
        "Download PDF",
        data=build_pdf_bytes(clean_notes, tasks),
        file_name=f"{stem}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    col4.download_button(
        "Download DOCX",
        data=build_docx_bytes(clean_notes, tasks),
        file_name=f"{stem}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )


def main() -> None:
    _init_state()

    st.sidebar.header("Image Enhancement")
    st.session_state["enhancement_mode"] = st.sidebar.selectbox(
        "Enhancement Mode",
        options=["binary", "balanced", "none"],
        index=1,  # Default to balanced
        help="Choose how to preprocess images for OCR: binary (high contrast), balanced (moderate), or none (original)"
    )

    st.title("Scribble to Digital")
    st.caption("Convert handwritten notes into cleaned text and structured tasks.")

    api_key = get_api_key_from_env()
    uploaded_file = st.file_uploader(
        "Upload a handwritten image",
        type=["jpg", "jpeg", "png"],
    )

    if not uploaded_file:
        st.info("Upload an image to begin.")
        return

    preview_image, image_array = _read_image(uploaded_file)
    st.session_state["image_name"] = uploaded_file.name

    left_col, right_col = st.columns([1.15, 1])
    with left_col:
        st.image(preview_image, caption=uploaded_file.name, use_container_width=True)

        if st.button("Run OCR", use_container_width=True):
            with st.spinner("Extracting text from the image..."):
                try:
                    ocr_service = OCRService(enhancement_mode=st.session_state["enhancement_mode"])
                    st.session_state["processed_preview"] = ocr_service.preprocess(image_array)
                    ocr_text = ocr_service.extract_text(image_array)
                except Exception as exc:
                    st.error(f"OCR failed: {exc}")
                else:
                    st.session_state["ocr_text"] = ocr_text
                    st.session_state["clean_notes"] = ""
                    st.session_state["tasks"] = []
                    st.session_state["model"] = ""
                    st.session_state["raw_response"] = ""
                    if ocr_text.strip():
                        st.success("OCR completed.")
                    else:
                        st.warning(
                            "OCR ran, but no text was detected. Try a clearer image with better lighting or higher contrast."
                        )

        if st.session_state["processed_preview"] is not None:
            st.subheader("Preprocessed Image")
            st.image(
                st.session_state["processed_preview"],
                caption="Image variant sent to OCR",
                use_container_width=True,
                clamp=True,
            )

    with right_col:
        st.subheader("Raw OCR Text")
        st.text_area(
            "OCR Result",
            height=260,
            key="ocr_text",
        )

        can_process = bool(st.session_state["ocr_text"].strip())
        if st.button(
            "Convert to Smart Digital Output",
            disabled=not can_process,
            use_container_width=True,
        ):
            if not api_key:
                st.error("Missing Gemini API key.")
            else:
                with st.spinner("Cleaning notes and extracting tasks..."):
                    result = None
                    try:
                        result = AIService(api_key).process(st.session_state["ocr_text"])
                    except FallbackProcessingError as exc:
                        st.warning(str(exc))
                        result = exc.fallback_result
                    except QuotaExceededError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        error_detail = str(exc)
                        st.error(f"AI processing failed: {error_detail}")
                        # Show helpful info
                        with st.expander("Troubleshooting"):
                            st.write("**Possible causes:**")
                            if len(st.session_state["ocr_text"]) > 20000:
                                st.write("- Text is too long (exceeds 20,000 characters). Try with a shorter section.")
                            st.write("- API quota exceeded or rate limited")
                            st.write("- Network/connectivity issue")
                            st.write(f"- Error detail: {error_detail}")
                    if result is not None:
                        st.session_state["clean_notes"] = result["clean_notes"]
                        st.session_state["tasks"] = result["tasks"]
                        st.session_state["model"] = result["model"]
                        st.session_state["raw_response"] = result.get("raw_response", "No raw response")

    if st.session_state["clean_notes"] or st.session_state["tasks"]:
        st.divider()
        output_col, task_col = st.columns([1.4, 0.8])
        with output_col:
            st.subheader("Clean Notes")
            st.write(st.session_state["clean_notes"] or "No notes generated.")
            if st.session_state["model"]:
                st.caption(f"Model: {st.session_state['model']}")
        with task_col:
            st.subheader("Tasks")
            tasks = st.session_state["tasks"]
            if tasks:
                for item in tasks:
                    st.write(f"- {item}")
            else:
                st.write("No tasks detected.")

        st.subheader("Downloads")
        _render_downloads(
            st.session_state["clean_notes"],
            st.session_state["tasks"],
            st.session_state["image_name"],
        )

        # Debug section
        with st.expander("Debug: AI Response"):
            st.text_area("Raw AI Response", st.session_state.get("raw_response", "No response yet"), height=200)


if __name__ == "__main__":
    main()