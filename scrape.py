import os
import sys
import warnings
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning)

# ---------- CONFIG ----------
YOUR_WEBSITE_URL = os.getenv("YOUR_WEBSITE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "portfolio-index")
EMBEDDING_DIM = 384


# ---------- SCRAPER ----------
def scrape_dynamic_react_site(url):
    print(f"Scraping site with Playwright: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html_content = page.content()
        browser.close()
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'noscript', 'nav', 'footer']):
        tag.decompose()
    return soup.get_text(separator='\n', strip=True)

# ---------- CHUNKING ----------
def chunk_text(text, chunk_size=300):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    chunks, current_chunk, current_words = [], [], 0
    for line in lines:
        words = line.split()
        if current_words + len(words) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_words = len(words)
        else:
            current_chunk.extend(words)
            current_words += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ---------- PINECONE ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = pc.list_indexes().names()

if INDEX_NAME in existing_indexes:
    info = pc.describe_index(INDEX_NAME)
    if info['dimension'] != EMBEDDING_DIM:
        print(f"⚠️ Index dimension mismatch. Deleting and recreating.")
        pc.delete_index(INDEX_NAME)
        # Wait a moment for deletion to propagate
        import time
        time.sleep(2)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {INDEX_NAME} (dim {EMBEDDING_DIM})")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Match your region
    )

index = pc.Index(INDEX_NAME)

# ---------- SCRAPE & UPSERT ----------
try:
    raw_text = scrape_dynamic_react_site(YOUR_WEBSITE_URL)
    chunks = chunk_text(raw_text)
    print(f"Extracted {len(chunks)} text chunks.")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks).tolist()

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk_{i}",
            "values": emb,
            "metadata": {"text": chunk}
        })

    batch_size = 100
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
        print(f"   Upserted batch {i//batch_size + 1}/{total_batches}")

    print(f"Successfully indexed {len(chunks)} chunks into Pinecone.")

except Exception as e:
    print(f"Indexing failed: {e}")
    sys.exit(1)

print("All done.")