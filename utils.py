import re
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def raw_text_to_markdown(text: str) -> str:
    """
    Convert raw PDF-extracted text into clean markdown format using heuristics.
    This ensures chunks stored in FAISS contain structured markdown for better retrieval.
    """
    lines = text.split("\n")
    markdown_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty separator lines
        if not line:
            i += 1
            continue

        # -------- HEADING DETECTION -------- #
        is_heading = False
        if 1 < len(line) < 100 and not line.endswith(".") and not line.endswith(":"):
            words = line.split()
            # All caps (3+ words)
            if len(words) >= 2 and all(w.isupper() for w in words if len(w) > 2):
                is_heading = True
            # Title case (first letter of each word capitalized, 2+ words)
            elif len(words) >= 2 and all(
                w[0].isupper() for w in words if w[0].isalpha()
            ):
                # Exclude regular sentences (ending with punctuation)
                if not re.search(r"[.!?:;]$", line):
                    is_heading = True

        if is_heading:
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

            # Section numbers like "4.3", "6.5" indicate level-3 headings
            if re.match(r"^\d+[\.\d]*\s", line):
                markdown_lines.append(f"### {line}")
            else:
                markdown_lines.append(f"## {line}")
            i += 1
            continue

        # -------- BULLET LIST DETECTION -------- #
        if re.match(r"^[\-\*]\s", line) or re.match(r"^\d+[\.\)]\s", line):
            markdown_lines.append(line)
            i += 1
            continue

        # -------- SECTION LABEL DETECTION (e.g. "Posology:", "Approval:") -------- #
        if re.match(r"^[A-Z][a-zA-Z\s]+:", line) and len(line) < 60:
            markdown_lines.append(f"**{line}**")
            i += 1
            continue

        # -------- NORMAL PARAGRAPH -------- #
        para = []
        while i < len(lines):
            current = lines[i].strip()
            if not current:
                i += 1
                break
            # Stop if next line looks like a heading
            if (
                1 < len(current) < 100
                and not current.endswith(".")
                and not current.endswith(":")
            ):
                words = current.split()
                if len(words) >= 2 and (
                    all(w.isupper() for w in words if len(w) > 2)
                    or (
                        all(w[0].isupper() for w in words if w[0].isalpha())
                        and not re.search(r"[.!?:;]$", current)
                    )
                ):
                    break
            para.append(current)
            i += 1

        if para:
            markdown_lines.append(" ".join(para))
            markdown_lines.append("")  # blank line after paragraph

    return "\n".join(markdown_lines).strip()


def load_and_chunk_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file, convert text to markdown, then split into chunks.
    Returns a list of Document objects with markdown-formatted content.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Convert each page's raw text to markdown format
    for doc in documents:
        doc.page_content = raw_text_to_markdown(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
