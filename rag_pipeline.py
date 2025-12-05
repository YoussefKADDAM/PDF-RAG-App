from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama


def extract_pdf_text(pdf_file):
    """Extract raw text from a PDF file."""
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    return text.strip()


def split_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into semantic chunks."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks):
    """Create the FAISS vector store using Nomic embeddings."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.from_texts(chunks, embedding=embeddings)


def load_llm(model_name="llama3.1:8b"):
    """Load the Ollama LLM."""
    return Ollama(model=model_name)


def summarize_text(llm, text, mode="short"):
    """Generate summaries: short, detailed, or bullet points."""

    if mode == "short":
        prompt = f"Give a concise 3–4 sentence summary of this document:\n\n{text}"

    elif mode == "detailed":
        prompt = f"Give a detailed multi-paragraph summary of this document:\n\n{text}"

    elif mode == "bullet":
        prompt = f"Summarize this document into clear bullet points:\n\n{text}"

    else:
        raise ValueError("Unknown summary mode.")

    return llm(prompt)
