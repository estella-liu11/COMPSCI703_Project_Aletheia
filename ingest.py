import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- System Configuration ---
# It's better to use environment variables for security,
# but I'm keeping your key here as requested for immediate use.
os.environ["PINECONE_API_KEY"] = "pcsk_vwEEW_Nm8c5qsyQ4dHN1itWQfepSATHBk2gnoSMyGzpvQ5abry7j7hTCyS9gGWgNkJshw"
INDEX_NAME = "aletheia-legal-db"

def start_ingestion():
    print("--- Starting Knowledge Ingestion Pipeline ---")

    # 1. Initialize the Embedding Model (Converting text to vectors)
    # This model outputs 384 dimensions - ensure your Pinecone index matches this.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Load Documents from Directory
    try:
        # This will load all PDFs inside the 'DATA-library' folder
        loader = PyPDFDirectoryLoader("DATA-library")
        data = loader.load()
        print(f"Successfully loaded {len(data)} pages from documents.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 3. Document Chunking (NLP Chunking)
    # We split long documents into 500-character chunks with a 50-character overlap
    # This allows the retrieval system to find specific clauses accurately.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(data)
    print(f"Text splitting complete: Created {len(docs)} knowledge chunks.")

    # 4. Upsert to Pinecone Vector Database
    print("Connecting to Pinecone and uploading vectors...")
    try:
        # Note: This step sends your text + vectors to the cloud.
        PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=INDEX_NAME
        )
        print("✅ Ingestion Successful! Your Pinecone index is now populated.")
    except Exception as e:
        print(f"Upload failed. Please check if index name is correct and dimensions are set to 384. Error: {e}")

if __name__ == "__main__":
    start_ingestion()