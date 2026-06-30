import os
import warnings
import sys
from dotenv import load_dotenv

# 1. Clear out internal package syntax warnings completely
warnings.filterwarnings("ignore", category=SyntaxWarning)

from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import chromadb
# Import Settings to natively kill telemetry on initialization
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI

load_dotenv()

# ---------- CONFIGURATION ----------
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
YOUR_WEBSITE_URL = os.getenv("YOUR_WEBSITE_URL")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global dictionary to track conversation history per sessionId
conversations_memory = {}
MAX_HISTORY_TURNS = 6  # Kept concise for speed and focus

# ---------- 1. DYNAMIC REACT SCRAPER ----------
def scrape_dynamic_react_site(url):
    print(f"🌐 Headless browser parsing React application: {url}")
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

# ---------- 2. CONTEXTUAL CHUNKING ----------
def chunk_text(text, chunk_size=300):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    chunks = []
    current_chunk = []
    current_words = 0
    
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

# ---------- INITIALIZATION SEQUENCE ----------
print("--- 🤖 Initializing RAG Backend Engine ---")
try:
    raw_text = scrape_dynamic_react_site(YOUR_WEBSITE_URL)
    site_chunks = chunk_text(raw_text)
    print(f"✅ Ingested {len(site_chunks)} web layouts data chunks successfully.")
except Exception as e:
    print(f"❌ Initial site context recovery failed: {e}")
    sys.exit(1)

# Initialize ChromaDB with telemetry explicitly turned off in Settings
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

try:
    chroma_client.delete_collection(name="dynamic_portfolio")
except Exception:
    pass

collection = chroma_client.get_or_create_collection(
    name="dynamic_portfolio",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

collection.add(documents=site_chunks, ids=[f"live_chunk_{i}" for i in range(len(site_chunks))])
print("✅ Vector index completely loaded into memory. Server engine active.\n")

# ---------- 3. CHAT API ROUTE WITH MEMORY ----------
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        if not data or 'chatInput' not in data:
            return jsonify({"text": "Missing parameter payload inputs."}), 400
            
        user_query = data['chatInput']
        session_id = data.get('sessionId', 'anonymous') 
        
        if session_id not in conversations_memory:
            conversations_memory[session_id] = []
            
        # 1. Safe context retrieval
        results = collection.query(query_texts=[user_query], n_results=3)
        retrieved_docs = []
        
        if results and isinstance(results, dict):
            docs_list = results.get('documents')
            if docs_list and len(docs_list) > 0 and docs_list[0] is not None:
                retrieved_docs = docs_list[0]
                
        context = "\n\n".join(retrieved_docs) if retrieved_docs else "[No explicit context found]"
        
        # 2. Strict system instructions for short, straightforward replies
        system_prompt = (
            "You are an intelligent assistant trained on the structure of this portfolio website. "
            "Answer the user query based purely on the website context provided below. "
            "CRITICAL: Be extremely concise, direct, and straightforward. Provide your answer in 1 to 3 sentences maximum. "
            "Avoid fluff, introductory remarks, or long bulleted lists. "
            "If the information is not present in the context, state exactly: 'I don't have that information on the website.'\n\n"
            f"Website Context:\n{context}"
        )
        
        api_messages = [{"role": "system", "content": system_prompt}]
        
        for past_message in conversations_memory[session_id]:
            api_messages.append(past_message)
            
        api_messages.append({"role": "user", "content": user_query})
        
        # 3. Call OpenRouter Gateway using Nemotron
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=api_messages,
            max_tokens=150,       # Caps lengths forcefully
            temperature=0.3       # Low temperature keeps answers sharp and factual
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # 4. Commit to Session Memory
        conversations_memory[session_id].append({"role": "user", "content": user_query})
        conversations_memory[session_id].append({"role": "assistant", "content": ai_response})
        
        if len(conversations_memory[session_id]) > MAX_HISTORY_TURNS:
            conversations_memory[session_id] = conversations_memory[session_id][-MAX_HISTORY_TURNS:]
        
        return jsonify({
            "output": ai_response, 
            "text": ai_response
        })
        
    except Exception as e:
        print(f"❌ Session handler failed context parsing execution: {e}")
        return jsonify({"text": "Internal error handling text frame transformation."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)