from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env

import tempfile
import re
import json
import queue
import threading
import time
import logging
import warnings
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from datasets import Dataset
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

logger = logging.getLogger("summary-generator")

def setup_logging():
    """Configure logging so INFO-level progress is visible no matter how the
    app is launched (python app.py OR python -m uvicorn app:app)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

setup_logging()

try:
    from transformers import TextIteratorStreamer
except ImportError:
    TextIteratorStreamer = None

# langchain-community is being sunset (https://github.com/langchain-ai/langchain-community/issues/674).
# The package still works, but its import emits a DeprecationWarning. Suppress it
# until this codebase is migrated to the standalone integration packages
# (langchain-pdf for PyPDFLoader, langchain-faiss for FAISS, ...).
warnings.filterwarnings(
    "ignore",
    message=r"`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)

# bitsandbytes 0.49.x emits FutureWarnings from torch internals
# (torch._check_is_size). These are harmless and come from upstream, so keep
# the server logs clean by filtering them out.
warnings.filterwarnings(
    "ignore",
    message=r".*_check_is_size.*",
    category=FutureWarning,
)

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# HuggingFace imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

# In-memory store for uploaded document vectorstores and metadata
DOCUMENT_STORE: dict = {}

# ---------- Configuration ----------

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

# Auto-detect GPU count and device
DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE = "cuda" if DEVICE_COUNT > 0 else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Cached model instances (load once at startup)
_embeddings_model: Optional[HuggingFaceEmbeddings] = None
_llm_pipeline: Optional[HuggingFacePipeline] = None

if DEVICE == "cuda":
    print(f"✅ {DEVICE_COUNT} GPU(s) detected:")
    for i in range(DEVICE_COUNT):
        props = torch.cuda.get_device_properties(i)
        print(
            f"   GPU {i}: {torch.cuda.get_device_name(i)} | "
            f"{props.total_memory / 1024**3:.1f} GB | "
            f"Compute Capability: {props.major}.{props.minor}"
        )
else:
    print("⚠️  No GPU detected. Running on CPU (this will be slow).")


# ---------- Startup Event: Preload Models ----------


async def load_models():
    """Preload embedding and LLM models at startup."""

    global _embeddings_model, _llm_pipeline, config, REGULATORY_VECTORSTORE

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        pipeline,
        BitsAndBytesConfig,
    )

    # -------------------------------------------------
    # 1. Load embedding model first
    # -------------------------------------------------

    print("⏳ Loading embedding model...")

    _embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": DEVICE,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    print(f"✅ Embedding model loaded: " f"{EMBEDDING_MODEL_NAME}")

    # -------------------------------------------------
    # 2. Load FAISS vector store
    # -------------------------------------------------

    try:
        REGULATORY_VECTORSTORE = FAISS.load_local(
            "knowledge_base",
            _embeddings_model,
            allow_dangerous_deserialization=True,
        )

        print("✅ Regulatory knowledge base loaded")

    except Exception as e:
        print(f"⚠️ Could not load regulatory knowledge base: {e}")

        REGULATORY_VECTORSTORE = None

    # -------------------------------------------------
    # 3. Load model config
    # -------------------------------------------------

    print("⏳ Loading LLM config...")

    config = AutoConfig.from_pretrained(
        LLM_MODEL_NAME,
        token=HF_TOKEN,
    )

    # Do NOT force `tie_word_embeddings = False`. Gemma checkpoints ship tied
    # input/output embeddings and contain no `lm_head.weight`; forcing the tie
    # off makes transformers allocate and randomly initialize lm_head (the
    # "lm_head.weight | MISSING" load report), which silently corrupts every
    # generated token. Respect whatever the checkpoint declares instead.

    # -------------------------------------------------
    # 4. Load tokenizer
    # -------------------------------------------------

    print("⏳ Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        token=HF_TOKEN,
    )

    print(f"✅ Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"✅ Tokenizer eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"✅ Tokenizer pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"✅ Tokenizer bos_token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("⚠️ pad_token was None, set to eos_token")

    # Decoder-only models must pad on the LEFT: right padding puts pad tokens
    # between the prompt and the first generated token, so batched inference
    # continues from garbage instead of the prompt.
    tokenizer.padding_side = "left"

    # -------------------------------------------------
    # 5. Configure model loading
    # -------------------------------------------------

    print("⏳ Loading LLM model...")

    model_kwargs = {}

    if DEVICE == "cuda":

        model_kwargs = {
            "device_map": "auto",
        }

        # Use 4-bit quantization for RTX 4080 12GB
        if os.getenv("LOAD_IN_4BIT", "1") == "1":

            print("⚡ Loading model in 4-bit mode")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model_kwargs["quantization_config"] = quantization_config

        else:

            print("⚠️ Loading model in FP16 mode")

            model_kwargs["torch_dtype"] = torch.float16

    # -------------------------------------------------
    # 6. Load model
    # -------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        config=config,
        token=HF_TOKEN,
        **model_kwargs,
    )

    # Only needed if device_map wasn't used
    if DEVICE == "cuda" and "device_map" not in model_kwargs:
        model = model.to(DEVICE)

    # Verify tie_word_embeddings is respected (should be True for Gemma)
    print(f"✅ Model tie_word_embeddings: {getattr(config, 'tie_word_embeddings', 'N/A')}")
    print(f"✅ Model has lm_head: {hasattr(model, 'lm_head')}")
    if hasattr(model, 'lm_head'):
        print(f"✅ lm_head weight shape: {model.lm_head.weight.shape}")

    # -------------------------------------------------
    # 7. Show device information
    # -------------------------------------------------

    if DEVICE == "cuda":

        if hasattr(model, "hf_device_map"):

            devices_used = set(model.hf_device_map.values())

            print(
                f"✅ Model using " f"{len(devices_used)} device(s): " f"{devices_used}"
            )

        else:

            print(f"✅ Model loaded on: {DEVICE}")

    # -------------------------------------------------
    # 8. Create generation pipeline
    # -------------------------------------------------

    # Explicit generation defaults:
    #  - If max_new_tokens stays unset the pipeline falls back to its 256-token
    #    default (which truncated long summaries mid-sentence). It is supplied
    #    per-call via `.bind(pipeline_kwargs=...)` on the map/reduce chains.
    #  - Clearing it on the generation_config is enough: the pipeline inherits
    #    that config, so no `max_length` kwarg is needed at creation time. Passing
    #    it there also triggered "Passing `generation_config` together with
    #    generation-related arguments is deprecated".
    #  - clean_up_tokenization_spaces=False silences the BPE tokenizer warning.
    #  - Ensure eos_token_id and pad_token_id are set on generation_config so
    #    the model stops cleanly and doesn't emit pad/eos tokens mid-generation.
    model.generation_config.max_length = None
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.generation_config, "bos_token_id") and model.generation_config.bos_token_id is None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )

    # The transformers pipeline warns "use a dataset" when its internal
    # call_count exceeds 10 on GPU — a heuristic that misfires against our
    # explicit batch_size usage and spams the logs on every batch call. Reset
    # the counter before each real call so the heuristic never trips.
    _pipeline_call = pipe.__call__

    def _bounded_call(*args, **kwargs):
        pipe.call_count = 0
        return _pipeline_call(*args, **kwargs)

    pipe.__call__ = _bounded_call

    _llm_pipeline = HuggingFacePipeline(
        pipeline=pipe,
    )

    print(f"✅ LLM model loaded successfully: " f"{LLM_MODEL_NAME}")

    if DEVICE == "cuda":

        print(f"GPU: " f"{torch.cuda.get_device_name(0)}")

        print(f"Allocated VRAM: " f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        print(f"Reserved VRAM: " f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB")


# ---------- Lifespan & FastAPI App ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.

    Load the embedding and LLM models once at startup (replacing the deprecated
    `@app.on_event("startup")` mechanism) and then yield control to the app.
    Any cleanup logic can be added after the yield if needed.
    """
    await load_models()
    yield


app = FastAPI(
    title="PDF Summary Generator API",
    description="Upload a PDF and ask questions about its content using LangChain + HuggingFace",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Helper Functions ----------


def get_embeddings():
    """Return the cached embedding model (loaded at startup)."""
    global _embeddings_model

    if _embeddings_model is None:
        raise RuntimeError(
            "Embedding model not loaded. The startup event may have failed. "
            "Check the server logs for details."
        )

    return _embeddings_model


def get_llm() -> HuggingFacePipeline:
    """Return the cached HuggingFace LLM pipeline (loaded at startup)."""
    global _llm_pipeline

    if _llm_pipeline is None:
        raise RuntimeError(
            "LLM model not loaded. The startup event may have failed. "
            "Check the server logs for details."
        )

    return _llm_pipeline


_STOPWORDS = frozenset(
    """
    about above after again against all also am among an and any are as at be
    because been before being below between both but by can cannot could did do
    does doing down during each few for from further had has have having he her
    here hers him his how i if in into is it its just me more most my no nor not
    now of off on once only or other our ours out over own same she should so
    some such than that the their theirs them then there these they this those
    through to too under until up very was we were what when where which while
    who whom why will with would you your yours describe explain summary
    summarize summarise tell detail give know want please help me my document
    pdf file uploaded what's whats is are does did do can could would will
    """.split()
)

# Relevance-score thresholds (cosine similarity from FAISS, 0..1, higher = more related).
# Measured with all-MiniLM-L6-v2: document-related questions score >= ~0.15, clearly
# off-topic questions score strongly negative (<= -0.8). Below REJECT the question is
# clearly unrelated; between REJECT and GRAY it is treated as off-topic UNLESS the
# question shares meaningful vocabulary with the retrieved document chunk.
OFF_TOPIC_REJECT_SCORE = 0.05
OFF_TOPIC_GRAY_SCORE = 0.25

NOT_RELATED_MESSAGE = (
    "This question is not related to the uploaded document. I can only "
    "help with questions about the content of the uploaded document."
)


def _question_keywords(question: str) -> list[str]:
    """Extract meaningful content words from a question (stopwords removed)."""
    words = re.findall(r"[A-Za-z]{3,}", question.lower())
    return [w for w in words if w not in _STOPWORDS]


def _lexical_match(question: str, docs: list[Document]) -> bool:
    """Return True if at least one question keyword appears in the retrieved
    document chunks (used as a secondary signal in the off-topic gate)."""
    keywords = _question_keywords(question)
    if not keywords:
        return True  # nothing meaningful to match — do not reject on this alone
    doc_text = " ".join(d.page_content for d in docs).lower()
    return any(w in doc_text for w in keywords)


def clean_query_answer(answer: str, question: str) -> str:
    """Strip common LLM artifacts from a query answer."""
    if not answer:
        return answer

    # Remove leading "Answer:" / "Response:" / "The answer is:" prefixes
    answer = re.sub(
        r"^\s*(?:Answer|Response|The answer is|Based on .*?|Content|Section Summary"
        r"|Paragraph Summary|Final Markdown Summary)\s*[:\-]?\s*",
        "",
        answer,
        count=1,
        flags=re.IGNORECASE,
    )

    # Remove echoed question at the start (Mistral often repeats the Q)
    q_stripped = question.strip().rstrip("?.")
    if q_stripped:
        pattern = re.compile(
            re.escape(q_stripped) + r"\s*[:\-]?\s*",
            re.IGNORECASE,
        )
        m = pattern.match(answer)
        if m:
            answer = answer[m.end() :]

    # Remove source labels the model might echo back mid-answer
    answer = re.sub(
        r"\s*\[(Uploaded Document:[^\]]*|FDA/EMA Regulatory Guidance)\]\s*", " ", answer
    )

    # Cut any leaked prompt labels the model echoes back mid-text, e.g. a stray
    # "Content:" or "Section Summary:" line after a token-limit stop.
    for label in (
        "Content",
        "Section Summary",
        "Paragraph Summary",
        "Final Markdown Summary",
        "Final Comprehensive Markdown Summary",
    ):
        m = re.search(rf"\n\s*{re.escape(label)}\s*[:\-]", answer, flags=re.IGNORECASE)
        if m and re.search(r"\S", answer[m.end():]):
            answer = answer[: m.start()].rstrip()
            break

    # The model sometimes keeps chatting in the Q&A format and appends an extra
    # "Question: ... Answer: ..." block. Cut everything from the start of that
    # trailing self-generated question so the answer ends cleanly.
    m = re.search(
        r"\n\s*(?:Question|Q)\s*[:\-]|^\s*(?:Question|Q)\s*[:\-]",
        answer,
        flags=re.IGNORECASE,
    )
    if m and re.search(
        r"\n\s*(?:Answer|A)\s*[:\-]", answer[m.end():], flags=re.IGNORECASE
    ):
        answer = answer[: m.start()].rstrip()

    # Normalize whitespace so the rendered markdown is clean
    answer = re.sub(r"[ \t]+\n", "\n", answer)  # trailing spaces per line
    answer = re.sub(r"\n{3,}", "\n\n", answer)  # max one blank line
    answer = re.sub(r"[ \t]{2,}", " ", answer)  # collapse double spaces in text

    return answer.strip()


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
        chunk_size=2000,
        chunk_overlap=100,
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


def create_vectorstore(chunks: List[Document]) -> FAISS:
    """Create a FAISS vector store from document chunks.

    Cosine distance is used so relevance scores are cosine similarities in
    [0, 1], which the off-topic gate relies on.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return vectorstore


REDUCE_GROUP_SIZE = 6
MAP_MAX_NEW_TOKENS = 512
REDUCE_MAX_NEW_TOKENS = 1024

# Reduce stops once only this many section summaries remain; the final summary
# is built by assembling those sections, so nothing is lost to a giant final
# merge and the output can never be cut off at a token limit.
FINAL_SECTION_COUNT = 6


def _extract_generated_text(item):
    """Recursively pull the generated text out of any pipeline result shape.

    Batched text-generation can return items as a plain string, a dict
    {"generated_text": ...}, or a list-wrapped dict [{"generated_text": ...}]
    depending on the transformers version. A list-wrapped dict would
    otherwise fall through to str(item) and leak the raw JSON `[{'generated_text': ...}]`
    into summaries. This normalizes every shape to the plain text.
    """
    if isinstance(item, dict):
        return item.get("generated_text", "")
    if isinstance(item, list):
        if not item:
            return ""
        return _extract_generated_text(item[0])
    return str(item)


def _normalize_summary(text: str) -> str:
    """If a model output is a serialized `[{'generated_text': ...}]` repr,
    unwrap it back to the plain text instead of displaying raw JSON."""
    stripped = text.strip()
    if stripped.startswith("[{") and stripped.endswith("}]"):
        try:
            parsed = json.loads(stripped)
            extracted = _extract_generated_text(parsed)
            if isinstance(extracted, str) and extracted.strip():
                return extracted
        except Exception:
            pass
    return text


def _strip_summary_fluff(text: str) -> str:
    """Remove the boilerplate small local models append around summaries:
    notes/sign-offs to the reader and self-generated Q&A or glossary sections
    that are formatting artifacts, not document content."""
    if not text:
        return text

    # 1) Cut trailing notes, sign-offs, and meta-commentary addressed to the
    #    reader ("Please let me know...", "Best regards...", explanations of the
    #    summarizer's own role). These are never part of the document.
    markers = (
        "note: please let me know",
        "please let me know if you need",
        "please let me know if any",
        "please let me know what",
        "please feel free to reach out",
        "let me know if you need",
        "let me know if anything",
        "if you have any questions",
        "if you need any modifications",
        "thank you for your time",
        "thank you for your patience",
        "thanks for your time",
        "thanks for reading",
        "best regards",
        "warm regards",
        "kind regards",
        "with regards",
        "i am looking forward to your feedback",
        "i look forward to your feedback",
        "i hope this meets your expectations",
        "i hope this answer meets",
        "i hope everything is clear",
        "happy to help",
        "hope this helps",
        "as an ai",
        "as a language model",
        "human expert technical document summarizer",
    )
    low = text.lower()
    cut = len(text)
    for marker in markers:
        idx = low.find(marker)
        if idx != -1 and idx < cut:
            cut = idx
    text = text[:cut].rstrip()

    # 2) Drop self-generated Q&A lines: "... ?  Answer: ..." or a lone
    #    "Answer: ..." line — the model invents these, the document doesn't have them.
    text = re.sub(
        r"(?m)^[ \t]*[^\n]*\?\s*(?:Answer|Ans\.?)\s*[:\-][^\n]*\n?",
        "",
        text,
    )
    text = re.sub(r"(?m)^[ \t]*(?:Answer|Response)\s*[:\-]\s*[^\n]*\n?", "", text)

    # 3) Drop model-generated heading blocks like:
    #    "Here are the answers to the following questions:" / "Here are the key terms and definitions:"
    #    Everything from such a header up to the next markdown heading (or EOF)
    #    is the model's own scaffolding, not document content.
    text = re.sub(
        r"(?im)^[ \t]*(?:here are the answers[^\n]*|here are the key terms[^\n]*"
        r"|here is the answer[^\n]*|here are the questions[^\n]*"
        r"|here is a summary[^\n]*|here is the summary[^\n]*"
        r"|key terms and definitions[^\n]*|frequently asked questions[^\n]*"
        r"|questions and answers[^\n]*)[ \t]*\n(?![ \t]*#).*(?=\n[ \t]*#|\Z)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 4) Collapse whitespace left behind.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedupe_paragraphs(text: str) -> str:
    """Remove duplicated paragraphs/blocks. Small models repeat the same few
    sentences when summarizing overlapping or repeated content, so a near
    verbatim repeat is dropped, keeping the first occurrence."""
    if not text:
        return text

    blocks = re.split(r"\n\s*\n", text)
    seen: list[str] = []
    kept: list[str] = []
    for block in blocks:
        norm = re.sub(r"\W+", " ", block.lower()).strip()
        if not norm:
            continue
        # Long block contained in (or containing) an earlier block → repeat.
        dup = False
        for s in seen:
            smaller, larger = (
                (norm, s) if len(norm) <= len(s) else (s, norm)
            )
            if len(smaller) >= 40 and smaller in larger:
                dup = True
                break
        if dup:
            continue
        seen.append(norm)
        kept.append(block)
    return "\n\n".join(kept).strip()


def _dedupe_sentences(text: str) -> str:
    """Collapse verbatim repeated sentences even when they sit inside one
    flowing paragraph — how small models typically inflate a summary."""
    if not text:
        return text
    sentences = re.findall(r"[^.!?]+[.!?]*", text)
    seen: set[str] = set()
    kept: list[str] = []
    for s in sentences:
        norm = re.sub(r"\W+", " ", s.lower()).strip()
        if len(norm.split()) < 4 or norm not in seen:
            seen.add(norm)
            kept.append(s.strip())
    return " ".join(kept).strip()


def clean_summary_part(text: str) -> str:
    """Full cleaning pipeline for one generated summary section (map/reduce)."""
    if not text:
        return ""
    text = _normalize_summary(text)
    text = clean_query_answer(text, "")
    text = _strip_summary_fluff(text)
    text = _dedupe_paragraphs(text)
    text = _dedupe_sentences(text)
    return text


def _batched_generate(
    llm: HuggingFacePipeline,
    prompts: list[str],
    max_new_tokens: int,
    initial_batch: int,
    progress_cb=None,
) -> list[str]:
    """
    Generate continuations for many prompts using TRUE batched inference.

    Many prompts are sent in a single pipeline call (batch_size), so N chunks
    share the prefill and KV-setup overhead instead of each paying it. Threads
    firing concurrent calls at one model instance do NOT batch on a single GPU
    — they each allocate their own KV cache and thrash the device, which is the
    reason the threaded map phase was slow. Batch size shrinks adaptively on OOM.
    `progress_cb(done, total)` is invoked after every finished batch so callers
    can stream live progress back to the client.
    """
    outputs: list[str] = []
    batch = max(1, initial_batch)
    i = 0
    total = len(prompts)
    batch_num = 0
    t_start = time.time()

    logger.info(
        "Batched generate: %d prompts, max_new_tokens=%d, initial_batch=%d",
        total, max_new_tokens, batch,
    )

    while i < total:
        batch_prompts = prompts[i : i + batch]
        batch_num += 1
        t_batch = time.time()

        try:
            # Wrap the batch in a huggingface Dataset, then feed the texts.
            # `dataset["text"]` yields a Column (not a list), which the pipeline
            # cannot batch, so cast to a plain list before calling.
            dataset = Dataset.from_dict({"text": batch_prompts})
            result = llm.pipeline(
                list(dataset["text"]),
                batch_size=batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.12,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if batch > 1:
                    batch = max(1, batch // 2)
                    logger.warning(
                        "OOM at batch %d (prompt %d/%d): reducing batch size → %d",
                        batch_num, i, total, batch,
                    )
                    continue
                logger.error(
                    "OOM at batch size 1 (prompt %d/%d): returning empty result", i, total
                )
                result = [{"generated_text": ""}] * len(batch_prompts)
            else:
                raise

        for item in result:
            outputs.append(_normalize_summary(_extract_generated_text(item)))
        i += len(batch_prompts)

        elapsed = time.time() - t_batch
        logger.info(
            "  batch %d/%d  done=%d/%d  batch_size=%d  elapsed=%.1fs",
            batch_num, max(1, (total + initial_batch - 1) // initial_batch),
            i, total, len(batch_prompts), elapsed,
        )

        if progress_cb is not None:
            progress_cb(min(i, total), total)

    total_elapsed = time.time() - t_start
    logger.info(
        "Batched generate complete: %d outputs, %.1fs total", len(outputs), total_elapsed
    )
    return outputs


def _merge_summaries(
    summaries: list[str],
    reduce_prompt: ChatPromptTemplate,
    llm: HuggingFacePipeline,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Merge groups of summaries into fewer summaries via a batched reduce pass."""
    groups = [
        summaries[i : i + REDUCE_GROUP_SIZE]
        for i in range(0, len(summaries), REDUCE_GROUP_SIZE)
    ]
    prompts = [
        reduce_prompt.format(summaries="\n\n".join(g))
        for g in groups
        if any(s.strip() for s in g)
    ]
    logger.info(
        "Merge pass: %d input summaries → %d groups → %d prompts",
        len(summaries), len(groups), len(prompts),
    )
    merged = _batched_generate(llm, prompts, max_new_tokens, batch_size)
    result = [clean_summary_part(m) for m in merged if m and m.strip()]
    logger.info("Merge pass result: %d summaries retained", len(result))
    return result


def _assemble_final_summary(summaries: list[str]) -> str:
    """Join the final summaries into one markdown document.

    Each block already carries its own thematic headings, so joining them under
    a single title yields one coherent whole-document summary — no "Section N"
    labels, and nothing is dropped or cut at a token limit.
    """
    cleaned = [s.strip() for s in summaries if s and s.strip()]

    if not cleaned:
        return "No summaries could be generated from the document chunks."

    if len(cleaned) == 1:
        return cleaned[0]

    result = "# Document Summary\n\n" + "\n\n".join(cleaned)
    # Light final pass: strip any cross-section fluff and duplicate blocks.
    result = _strip_summary_fluff(result)
    result = _dedupe_paragraphs(result)
    return result


def generate_summary(chunks: list[Document], emit=None) -> str:
    """
    Generate a summary using Map-Reduce with TRUE batched inference:

      MAP:   Each chunk is summarized into one short flowing paragraph. Chunks
             are grouped and the HF pipeline is invoked with a batch_size, so
             N chunks share prefill + KV-setup overhead. No threads — threading
             the same model never batches on a single GPU and is exactly why the
             old map phase was slow. Map outputs are capped at ~2-3 sentences so
             autoregressive decode finishes quickly (decode dominates runtime).

      REDUCE: Hierarchical and also batched. Groups of summaries are merged in
             batch into fewer summaries, which are merged again, until one final
             markdown summary remains. Every prompt stays small, so no single
             call approaches the context limit, and each level reaps the batch
             speedup.

    `emit` is an optional callback `emit(dict)` invoked with streaming events:
      {"type":"progress","done":..,"total":..}
      {"type":"summary_part","part":..,"index":..,"total":..}
      {"type":"reduce","level":..,"count":..}
      {"type":"final","summary":..}

    Returns a single markdown-formatted summary string.
    """

    llm = get_llm()

    # Conservative starting batches; _batched_generate shrinks them on OOM.
    initial_map_batch = 3 if DEVICE == "cuda" else 1
    initial_reduce_batch = 2 if DEVICE == "cuda" else 1

    t_start = time.time()
    logger.info(
        "=== Summary generation started: %d chunks, device=%s, "
        "map_batch=%d, reduce_batch=%d ===",
        len(chunks), DEVICE, initial_map_batch, initial_reduce_batch,
    )

    # ---------------- MAP PHASE ---------------- #
    map_prompt = ChatPromptTemplate.from_template(
        """You are an expert technical document summarizer.

Summarize the section below in your own words as ONE detailed, flowing paragraph.

Requirements:
- Write approximately 6-10 sentences (roughly 150-250 words) that capture ALL the
  key information present in the section.
- Cover the WHOLE section evenly, in the order the facts appear. Do not fixate on
  the first few lines and skip the rest.
- Preserve every important name, date, number, step, definition, requirement,
  approval, and technical detail. Do not leave important facts out.
- Do NOT use bullet points, dashes, or numbered lists — use flowing paragraphs.
- Never repeat the same fact or sentence twice.
- Output ONLY the factual content. Do NOT add any meta-commentary, notes or
  messages to the reader, greetings, sign-offs, or self-generated
  "Question: ... Answer: ..." blocks.
- Do NOT start with phrases like "Here is", "This section", "The following",
  or "I have summarized".

Document:

{context}

Paragraph Summary:"""
    )

    total = len(chunks)
    t_map_start = time.time()
    logger.info(
        "MAP PHASE START: %d chunks, max_new_tokens=%d, batch=%d",
        total, MAP_MAX_NEW_TOKENS, initial_map_batch,
    )

    map_prompts = [map_prompt.format(context=c.page_content) for c in chunks]

    def _on_progress(done: int, t: int):
        if emit:
            emit({"type": "progress", "done": done, "total": t})

    raw = _batched_generate(
        llm, map_prompts, MAP_MAX_NEW_TOKENS, initial_map_batch, progress_cb=_on_progress
    )

    summaries = []
    for idx, s in enumerate(raw):
        cleaned = clean_summary_part(s)
        if cleaned:
            summaries.append(cleaned)
            logger.info(
                "  chunk %d/%d summarized (%d chars)",
                idx + 1, total, len(cleaned),
            )
            if emit:
                emit(
                    {
                        "type": "summary_part",
                        "part": cleaned,
                        "index": idx,
                        "total": total,
                    }
                )
        else:
            logger.warning("  chunk %d/%d produced empty summary (dropped)", idx + 1, total)

    map_elapsed = time.time() - t_map_start
    kept = len(summaries)
    dropped = total - kept
    logger.info(
        "MAP PHASE COMPLETE: %d/%d chunks produced valid summaries (%d empty dropped), %.1fs",
        kept, total, dropped, map_elapsed,
    )

    if not summaries:
        logger.warning("MAP phase produced zero valid summaries — returning fallback")
        return "No summaries could be generated from the document chunks."

    if len(summaries) == 1:
        logger.info("Single summary produced — skipping reduce")
        if emit:
            emit({"type": "final", "summary": summaries[0]})
        return summaries[0]

    # ---------------- REDUCE PHASE (hierarchical, batched) ---------------- #
    # Neighboring chunks overlap, so the map summaries often repeat the same
    # facts. Dedupe before merging to keep the reduce prompt free of noise.
    pre_reduce = len(summaries)
    summaries = [
        s for s in (_dedupe_paragraphs(x) for x in summaries) if s
    ]
    if len(summaries) != pre_reduce:
        logger.info("Deduped map summaries: %d → %d", pre_reduce, len(summaries))

    if len(summaries) == 1:
        logger.info("Single unique summary produced — skipping reduce")
        if emit:
            emit({"type": "final", "summary": summaries[0]})
        return summaries[0]

    reduce_prompt = ChatPromptTemplate.from_template(
        """You are summarizing several related parts of the SAME document into ONE comprehensive, well-organized section summary.

Instructions:
- Organize the merged content under short THEMATIC markdown headings (##, ###) that describe the topics covered (e.g. "## Change Initiation", "## Approval and Sign-off"). Make the headings meaningful, not generic.
- Merge related facts together. If the same fact or sentence appears in more than
  one part, state it ONCE only — actively remove duplicated sentences and
  redundant phrasing. Do not paste the input parts back verbatim.
- Preserve ALL important names, dates, numbers, technical details, definitions, approvals, signatures, and requirements.
- Write in flowing paragraphs under each heading; use bullet lists ONLY when listing specific items (requirements, steps, etc.).
- Cover the content evenly and completely. Do not fixate on facts that appear
  first and drop the rest.
- Do NOT invent anything not present in the content, including "Key terms and definitions", "Frequently asked questions", "Here are the answers...", or any
  "Question: ... Answer: ..." blocks.
- Do NOT include any meta-commentary, notes to the reader, greetings, sign-offs, introductions, or conclusions (e.g. "This section covers...").
- Start directly with the first heading. Do NOT add a document title or the word "Summary".
- Output ONLY the markdown summary.

Content:
{summaries}
"""
    )

    t_reduce_start = time.time()
    logger.info(
        "REDUCE PHASE START: %d summaries, group_size=%d, batch=%d",
        len(summaries), REDUCE_GROUP_SIZE, initial_reduce_batch,
    )

    max_batch = initial_reduce_batch
    level = 1
    while len(summaries) > FINAL_SECTION_COUNT:
        t_level = time.time()
        logger.info(
            "REDUCE level %d: %d summaries → %d groups",
            level, len(summaries),
            (len(summaries) + REDUCE_GROUP_SIZE - 1) // REDUCE_GROUP_SIZE,
        )
        merged = _merge_summaries(
            summaries, reduce_prompt, llm, REDUCE_MAX_NEW_TOKENS, max_batch
        )

        # Defensive fallback: never return nothing on a merge failure
        if not merged or not any(s.strip() for s in merged):
            logger.warning(
                "REDUCE level %d produced empty result — falling back to raw summaries", level
            )
            result = _assemble_final_summary(summaries)
            total_elapsed = time.time() - t_start
            logger.info("=== Summary generation complete (fallback): %.1fs ===", total_elapsed)
            if emit:
                emit({"type": "final", "summary": result})
            return result

        summaries = merged
        logger.info(
            "REDUCE level %d complete: %d summaries remaining, %.1fs",
            level, len(summaries), time.time() - t_level,
        )
        if emit:
            emit({"type": "reduce", "level": level, "count": len(summaries)})
        level += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- FINAL ASSEMBLY ---------------- #
    # Stop at section level and join the sections into the final summary.
    # Everything the map/reduce steps produced is retained — nothing is dropped
    # by a last giant merge and the output cannot be cut at a token limit.
    result = _assemble_final_summary(summaries)

    total_elapsed = time.time() - t_start
    final_words = len(result.split())
    logger.info(
        "=== Summary generation complete: %d word output, %d map calls + %d reduce levels, %.1fs total ===",
        final_words, total, level - 1, total_elapsed,
    )
    if emit:
        emit({"type": "final", "summary": result})
    return result


# ---------- API Endpoints ----------


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "PDF Summary Generator API is running.",
        "docs": "/docs",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
    }


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract its text, create chunks, and store
    them in a FAISS vector store for later querying.
    Returns a session_id that can be used for subsequent queries.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # Load PDF and split into chunks
        chunks = load_and_chunk_pdf(tmp_path)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the PDF. The file may be empty or scanned.",
            )

        # Create vectorstore
        vectorstore = create_vectorstore(chunks)

        # Generate a session_id
        session_id = str(abs(hash(str(chunks[:5]))))

        # Store in memory
        DOCUMENT_STORE[session_id] = {
            "vectorstore": vectorstore,
            "filename": file.filename,
            "chunks": len(chunks),
            "chunk_objects": chunks,
        }

        return {
            "session_id": session_id,
            "filename": file.filename,
            "total_chunks": len(chunks),
            "message": f"PDF '{file.filename}' processed successfully with {len(chunks)} chunks.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)


@app.post("/query/")
async def query_pdf(
    session_id: str = Form(...),
    question: str = Form(...),
):
    """
    Ask a question about a previously uploaded PDF using its session_id.
    Uses RetrievalQA chain to answer based on the document chunks.
    """
    if session_id not in DOCUMENT_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"Session ID '{session_id}' not found. Please upload the PDF first.",
        )

    store_entry = DOCUMENT_STORE[session_id]
    vectorstore = store_entry["vectorstore"]
    filename = store_entry["filename"]

    try:
        t_retrieve = time.time()
        logger.info("Query received: '%s'", question[:120])

        uploaded_search = vectorstore.similarity_search_with_relevance_scores(
            question,
            k=5,
        )
        uploaded_docs = [doc for doc, _ in uploaded_search]
        uploaded_scores = [score for _, score in uploaded_search]

        # ---------- OFF-TOPIC GATE ----------
        # Only answer questions about the uploaded document. The regulatory
        # knowledge base is used solely to explain document content; it must
        # never enable answers for unrelated questions.
        best_score = max(uploaded_scores) if uploaded_scores else 0.0
        lexical_hit = _lexical_match(question, uploaded_docs)
        off_topic = best_score < OFF_TOPIC_REJECT_SCORE or (
            best_score < OFF_TOPIC_GRAY_SCORE and not lexical_hit
        )
        logger.info(
            "Relevance: best_score=%.3f lexical_hit=%s off_topic=%s",
            best_score, lexical_hit, off_topic,
        )

        if off_topic:
            logger.info("Question rejected as unrelated to the uploaded document")

            def no_context_stream():
                yield f"data: {json.dumps({'type': 'progress', 'status': 'checking'})}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'answer': NOT_RELATED_MESSAGE, 'uploaded_chunks': len(uploaded_docs), 'regulatory_chunks': 0, 'total_context_chunks': len(uploaded_docs)})}\n\n"

            return StreamingResponse(
                no_context_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        regulatory_docs = []

        if REGULATORY_VECTORSTORE is not None:
            regulatory_docs = REGULATORY_VECTORSTORE.similarity_search(
                question,
                k=5,
            )

        # Label the source of each context block so the model can distinguish
        # the uploaded document from FDA/EMA guidance.
        context_parts = [
            f"[Uploaded Document: {filename}]\n{doc.page_content}"
            for doc in uploaded_docs
        ]
        context_parts += [
            f"[FDA/EMA Regulatory Guidance]\n{doc.page_content}"
            for doc in regulatory_docs
        ]
        all_docs = uploaded_docs + regulatory_docs

        context = "\n\n".join(context_parts)
        retrieve_time = time.time() - t_retrieve
        logger.info(
            "Context retrieved: %d uploaded + %d regulatory = %d chunks | %.2fs",
            len(uploaded_docs), len(regulatory_docs), len(all_docs), retrieve_time,
        )

        prompt = ChatPromptTemplate.from_template(
            """You are an expert Regulatory Affairs assistant.

Context is provided from TWO clearly labeled sources:
1. [Uploaded Document: {filename}] — the document the user uploaded.
2. [FDA/EMA Regulatory Guidance] — a regulatory knowledge base (FDA/EMA guidance documents).

STRICT SCOPE RULES (MOST IMPORTANT):
- Answer questions ONLY about the content of the uploaded document.
- Use the FDA/EMA regulatory guidance ONLY as a reference to help explain or
  interpret regulatory terminology, requirements, or concepts that appear IN the
  uploaded document itself. Never answer a question from the guidance alone.
- If the question is NOT about the uploaded document (general knowledge,
  unrelated topics, or a different document), DO NOT answer the actual question.
  Say exactly: "This question is not related to the uploaded document. I can
  only help with questions about the content of the uploaded document."
- If the uploaded document does not contain the answer, say so clearly instead
  of guessing or inventing information.
- If the uploaded document and regulatory guidance contradict each other,
  explain the difference and indicate which source is the regulatory standard.

How to use the sources:
- ALWAYS base your answer first on the uploaded document.
- Use the regulatory guidance to clarify terminology and to add authoritative
  regulatory context, and refer to it when you use it (e.g. "per FDA guidance, ...").

Rules for your answer:
- Start directly with the answer. Do NOT repeat or rephrase the question.
- Do NOT include labels like "Answer:" or "Response:" at the beginning.
- Do NOT echo source labels such as "[Uploaded Document: ...]" or "[FDA/EMA Regulatory Guidance]" inside your answer.
- LEAD with a one-line direct answer stating the bottom line, THEN give the supporting details.
- Group the details under short markdown headings (##, ###) by topic.
- Write in short, plain sentences. Avoid repeating the same numbers or names.
- Use bullet points only when listing specific items (e.g. requirements, steps).
- ANSWER COMPLETELY: cover every part of the question in detail and do not stop
  early. If the question has multiple parts, answer each part explicitly.

Formatting rules for structured/form data:
- If the answer concerns a FORM, RECORD, or structured data from the uploaded document,
  REPRODUCE it in an organized layout instead of a plain paragraph:
  * Keep every labeled field (e.g. "Change Request No.", "Title", "Priority", "Initiating Department") as "Field: value".
  * Represent checked boxes (☑ / [X]) as "Yes / Checked" and unchecked (☐ / [ ]) as "No / Unchecked".
  * Do NOT merge separate rows or drop empty/blank fields.
- If the context contains tabular data (columns like From/To/Remarks, Yes/No/N/A), output it as a markdown TABLE.
- Preserve exact reference numbers (e.g. CCF No.), names, dates, times, and email addresses exactly as written.
- Separate distinct records with headings so each item is visually distinct.

Context:
{context}

Question: {question}

Answer:"""
        )

        messages = prompt.format_messages(
            context=context, question=question, filename=filename
        )
        prompt_str = messages[0].content

        def event_stream():
            q: "queue.Queue[str | None]" = queue.Queue()

            def emit(data: str):
                q.put(data)

            def run():
                try:
                    pipe = get_llm().pipeline
                    tokenizer = pipe.tokenizer

                    if TextIteratorStreamer is None:
                        raise RuntimeError(
                            "TextIteratorStreamer unavailable (transformers < 4.30)"
                        )

                    inputs = tokenizer(prompt_str, return_tensors="pt")
                    inputs = {k: v.to(pipe.device) for k, v in inputs.items()}

                    prompt_tokens = inputs["input_ids"].shape[1]
                    logger.info(
                        "Tokenizing done: prompt=%d tokens, generating on %s",
                        prompt_tokens, pipe.device,
                    )

                    streamer = TextIteratorStreamer(
                        tokenizer,
                        skip_special_tokens=True,
                        skip_prompt=True,
                    )

                    t_gen = time.time()

                    def generate():
                        pipe.model.generate(
                            **inputs,
                            streamer=streamer,
                            max_new_tokens=2048,
                            do_sample=False,
                            repetition_penalty=1.1,
                        )

                    thread = threading.Thread(target=generate, daemon=True)
                    thread.start()

                    emit(
                        f"data: {json.dumps({'type': 'progress', 'status': 'generating'})}\n\n"
                    )

                    full_answer = []
                    for token_text in streamer:
                        full_answer.append(token_text)
                        emit(
                            f"data: {json.dumps({'type': 'token', 'text': token_text})}\n\n"
                        )

                    thread.join()
                    gen_time = time.time() - t_gen

                    answer = clean_query_answer("".join(full_answer), question)
                    logger.info(
                        "Answer generated: %d tokens, %d chars | %.1fs",
                        len(full_answer), len(answer), gen_time,
                    )

                    emit(
                        f"data: {json.dumps({'type': 'final', 'answer': answer, 'uploaded_chunks': len(uploaded_docs), 'regulatory_chunks': len(regulatory_docs), 'total_context_chunks': len(all_docs)})}\n\n"
                    )
                except Exception as e:
                    logger.error("Query generation failed: %s", e)
                    emit(
                        f"data: {json.dumps({'type': 'error', 'detail': f'Error answering question: {str(e)}'})}\n\n"
                    )
                finally:
                    q.put(None)

            threading.Thread(target=run, daemon=True).start()

            while True:
                item = q.get()
                if item is None:
                    break
                yield item

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error("Query setup failed: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Error answering question: {str(e)}"
        )


@app.post("/summarize/")
async def summarize_pdf(session_id: str = Form(...)):
    """
    Generate a summary of the PDF content for the given session_id.

    Streams Server-Sent Events back to the client so progress and each section
    summary appear live while the map-reduce pipeline runs:
      {"type":"progress","done":..,"total":..}
      {"type":"summary_part","part":..,"index":..,"total":..}
      {"type":"reduce","level":..,"count":..}
      {"type":"final","summary":..}
      {"type":"error","detail":..}
    """
    if session_id not in DOCUMENT_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"Session ID '{session_id}' not found. Please upload the PDF first.",
        )

    store_entry = DOCUMENT_STORE[session_id]
    chunks = store_entry["chunk_objects"]
    filename = store_entry["filename"]

    def _emit(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def event_stream():
        # The blocking LLM work runs in a worker thread; this generator yields
        # each event as it is produced, so the client sees live progress.
        q: "queue.Queue[str | None]" = queue.Queue()

        def emit(payload: dict):
            q.put(f"data: {json.dumps(payload)}\n\n")

        def run():
            try:
                emit({"type": "start"})
                summary = generate_summary(chunks, emit=emit)
                emit({"type": "final", "summary": summary})
            except Exception as e:
                logger.error("Summary generation failed: %s", e)
                emit({"type": "error", "detail": f"Error generating summary: {str(e)}"})
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------- Main ----------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=0,
        timeout_graceful_shutdown=0,
    )
