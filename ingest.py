import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "aletheia-legal-db"


def start_ingestion():
    print("--- Starting Knowledge Ingestion Pipeline ---")

    # 1. Initialise the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Load PDFs
    try:
        loader = PyPDFDirectoryLoader("DATA-library")
        data = loader.load()
        print(f"✅ Loaded {len(data)} pages from PDFs")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # 3. Strip noisy PDF pages
    print(f"\n🧹 Cleaning noisy pages...")
    print(f"   Original pages: {len(data)}")

    cleaned_data = []
    for page in data:
        content = page.page_content.strip()

        if len(content) < 200:
            continue

        dot_ratio = content.count('.') / max(len(content), 1)
        if dot_ratio > 0.2:
            continue

        alphanumeric_count = sum(c.isalnum() for c in content)
        if alphanumeric_count / max(len(content), 1) < 0.3:
            continue

        cleaned_data.append(page)

    print(f"   After cleaning: {len(cleaned_data)} pages")
    print(f"   Removed: {len(data) - len(cleaned_data)} noisy pages")

    # 4. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )
    docs = splitter.split_documents(cleaned_data)
    print(f"✅ Created {len(docs)} chunks (chunk_size=800, overlap=80)")

    # 5. Upload to Pinecone
    print(f"\n📤 Uploading {len(docs)} chunks to Pinecone...")

    try:
        vector_store = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        print(f"✅ Successfully uploaded {len(docs)} chunks to Pinecone!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


if __name__ == "__main__":
    start_ingestion()