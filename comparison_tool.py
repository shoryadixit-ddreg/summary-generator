import streamlit as st
import pdfplumber
from docx import Document
import difflib
import io
import os
import tempfile
import subprocess

st.set_page_config(
    page_title="Human-Readable Visual Structure Diff",
    page_icon="🔎",
    layout="wide"
)

# Custom responsive CSS to render high-fidelity document layout containers
st.markdown("""
    <style>
    .scroll-container {
        background-color: #f1f5f9;
        padding: 30px;
        border-radius: 12px;
        max-height: 850px;
        overflow-y: auto;
        border: 1px solid #cbd5e1;
    }
    .document-page-view {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        padding: 50px 60px;
        margin: 0 auto 30px auto;
        max-width: 950px;
        font-family: "Times New Roman", Times, serif, sans-serif;
        color: #0f172a;
        line-height: 1.6;
        position: relative;
    }
    .page-divider {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 10px 0 25px 0;
        user-select: none;
    }
    .diff-line {
        margin: 0;
        padding: 2px 0;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .text-del {
        background-color: #fee2e2;
        color: #b91c1c;
        text-decoration: line-through;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 500;
    }
    .text-add {
        background-color: #dcfce7;
        color: #15803d;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .legend-banner {
        display: flex;
        gap: 25px;
        font-size: 14px;
        background: white;
        padding: 15px 25px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 25px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔎 Continuous Visual Structure Comparison Dashboard")
st.markdown("This tool compiles **all document pages into one continuous human-readable view**, keeping columns and spacing intact while formatting insertions and deletions inline.")

def convert_docx_to_pdf(docx_bytes) -> bytes:
    """Headless automated docx conversion to safely parse structural layout coordinates."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.docx")
        with open(input_path, "wb") as f:
            f.write(docx_bytes)
        try:
            subprocess.run([
                'libreoffice', '--headless', '--convert-to', 'pdf',
                '--outdir', temp_dir, input_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            output_pdf_path = os.path.join(temp_dir, "input.pdf")
            if os.path.exists(output_pdf_path):
                with open(output_pdf_path, "rb") as f:
                    return f.read()
        except Exception:
            return b""
    return b""

def extract_layout_lines(pdf_bytes) -> list:
    """Extracts text grouped by precise visual line heights and page sets to preserve alignment."""
    pages_data = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf_path = temp_pdf.name

    try:
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                # Group text blocks by their exact vertical coordinates on the layout sheet
                lines = page.extract_text(layout=True)
                page_lines = lines.splitlines() if lines else []
                pages_data.append(page_lines)
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            
    return pages_data

def compare_line_words(line_a, line_b):
    """Compares individual matching line strings word-by-word to create clean human-readable tags."""
    words_a = line_a.split()
    words_b = line_b.split()
    
    # If indentation padding spaces exist, preserve them for visual layout structure alignment
    leading_spaces = len(line_b) - len(line_b.lstrip())
    padding = "&nbsp;" * leading_spaces
    
    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    line_html = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            line_html.append(" ".join(words_a[i1:i2]))
        elif tag == 'replace':
            del_w = " ".join(words_a[i1:i2])
            add_w = " ".join(words_b[j1:j2])
            if del_w.strip(): line_html.append(f'<span class="text-del">{del_w}</span>')
            if add_w.strip(): line_html.append(f'<span class="text-add">{add_w}</span>')
        elif tag == 'delete':
            del_w = " ".join(words_a[i1:i2])
            if del_w.strip(): line_html.append(f'<span class="text-del">{del_w}</span>')
        elif tag == 'insert':
            add_w = " ".join(words_b[j1:j2])
            if add_w.strip(): line_html.append(f'<span class="text-add">{add_w}</span>')
            
    return f'{padding}{" ".join(line_html)}'

# Render Document Upload Interface
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Original File A (PDF/DOCX)", type=["pdf", "docx"], key="file_a")
with col2:
    file_b = st.file_uploader("Modified File B (PDF/DOCX)", type=["pdf", "docx"], key="file_b")

if file_a and file_b:
    bytes_a = file_a.read()
    bytes_b = file_b.read()
    file_a.seek(0)
    file_b.seek(0)

    # Convert word files automatically if loaded
    if file_a.name.lower().endswith('.docx'):
        with st.spinner("Processing structural mapping for Word Document A..."):
            bytes_a = convert_docx_to_pdf(bytes_a)
    if file_b.name.lower().endswith('.docx'):
        with st.spinner("Processing structural mapping for Word Document B..."):
            bytes_b = convert_docx_to_pdf(bytes_b)

    if not bytes_a or not bytes_b:
        st.error("Extraction error: Could not verify layout geometries. Please verify that LibreOffice is globally installed on your host system path.")
    else:
        with st.spinner("Analyzing text alignment coordinates across all pages simultaneously..."):
            pages_a = extract_layout_lines(bytes_a)
            pages_b = extract_layout_lines(bytes_b)

        # UI Legend Informer Block
        st.markdown("""
        <div class="legend-banner">
            <div><span class="text-del" style="text-decoration:line-through;">Struck-out Red Text</span> = Words removed from original document layout position.</div>
            <div><span class="text-add">Highlighted Green Text</span> = Words newly inserted directly next to them.</div>
        </div>
        """, unsafe_allow_html=True)

        total_pages = max(len(pages_a), len(pages_b))
        compiled_output_html = []
        
        # Start scroll window injection loop
        compiled_output_html.append('<div class="scroll-container">')

        for p_idx in range(total_pages):
            compiled_output_html.append(f'<div class="page-divider">--- Page {p_idx + 1} ---</div>')
            compiled_output_html.append('<div class="document-page-view">')
            
            lines_a = pages_a[p_idx] if p_idx < len(pages_a) else []
            lines_b = pages_b[p_idx] if p_idx < len(pages_b) else []
            
            # Match layout line elements inside this specific page
            page_matcher = difflib.SequenceMatcher(None, [l.strip() for l in lines_a], [l.strip() for l in lines_b])
            
            for tag, i1, i2, j1, j2 in page_matcher.get_opcodes():
                if tag == 'equal':
                    for idx in range(i1, i2):
                        # Preserve original spatial spacing formatting lines
                        leading_spaces = len(lines_b[j1 + (idx - i1)]) - len(lines_b[j1 + (idx - i1)].lstrip())
                        padding = "&nbsp;" * leading_spaces
                        compiled_output_html.append(f'<div class="diff-line">{padding}{lines_a[idx]}</div>')
                elif tag == 'replace':
                    sub_a = lines_a[i1:i2]
                    sub_b = lines_b[j1:j2]
                    for offset in range(max(len(sub_a), len(sub_b))):
                        line_a_txt = sub_a[offset] if offset < len(sub_a) else ""
                        line_b_txt = sub_b[offset] if offset < len(sub_b) else ""
                        inline_diff = compare_line_words(line_a_txt, line_b_txt)
                        compiled_output_html.append(f'<div class="diff-line">{inline_diff}</div>')
                elif tag == 'delete':
                    for line in lines_a[i1:i2]:
                        compiled_output_html.append(f'<div class="diff-line"><span class="text-del">{line}</span></div>')
                elif tag == 'insert':
                    for line in lines_b[j1:j2]:
                        compiled_output_html.append(f'<div class="diff-line"><span class="text-add">{line}</span></div>')
            
            compiled_output_html.append('</div>') # End Document Page Card

        compiled_output_html.append('</div>') # End Continuous Scroll Container

        # Render the full combined single-frame window layout safely onto the Streamlit UI
        st.markdown("### 📊 Continuous Document Redline View")
        st.markdown("".join(compiled_output_html), unsafe_allow_html=True)
else:
    st.info("💡 Pro-Tip: Drop two multi-page PDF or Word documents above. The engine handles all processing tasks simultaneously and displays the results in a single, continuous view.")
