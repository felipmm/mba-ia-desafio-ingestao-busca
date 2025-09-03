import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()
for k in ("GOOGLE_EMBEDDING_MODEL", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

def ingest_pdf():
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "pdf"
    documents = []

    for file in os.listdir(pdf_path):
        if file.endswith('.pdf'):
            file_path = os.path.join(pdf_path, file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, add_start_index=False).split_documents(documents)
    if not splits:
        raise SystemExit(0)

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL","models/embedding-001"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    store.add_documents(documents=enriched, ids=ids)

if __name__ == "__main__":
    ingest_pdf()