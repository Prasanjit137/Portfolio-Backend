import os
import warnings
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning)

# ---------- CONFIG ----------
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

logger.info(f"OPENROUTER_KEY: {'SET' if OPENROUTER_KEY else 'MISSING'}")
logger.info(f"PINECONE_API_KEY: {'SET' if PINECONE_API_KEY else 'MISSING'}")
logger.info(f"INDEX_NAME: {INDEX_NAME}")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

conversations_memory = {}
MAX_HISTORY_TURNS = 6

# ---------- PINECONE ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ---------- EMBEDDING (LOCAL MODEL) ----------
# The model is pre-downloaded during the Docker build, so this loads from local cache.
try:
    logger.info("Loading local embedding model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None

def get_embedding(text):
    """
    Generate embedding for a single text using the locally loaded SentenceTransformer.
    Returns a flat list of floats.
    """
    if embedding_model is None:
        raise RuntimeError("Embedding model is not available.")
    
    logger.info(f"Generating embedding locally for: {text[:50]}...")
    # encode() returns a numpy array; convert to list for Pinecone
    embedding = embedding_model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

# ---------- CHAT ENDPOINT ----------
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        if not data or 'chatInput' not in data:
            return jsonify({"text": "Missing parameter."}), 400

        user_query = data['chatInput']
        session_id = data.get('sessionId', 'anonymous')

        logger.info(f"Received query: {user_query} from session {session_id}")

        # 1. Embed user query locally
        query_emb = get_embedding(user_query)

        # 2. Query Pinecone
        results = index.query(vector=query_emb, top_k=3, include_metadata=True)
        retrieved_docs = []
        if results and results.get('matches'):
            for match in results['matches']:
                retrieved_docs.append(match['metadata']['text'])
        context = "\n\n".join(retrieved_docs) if retrieved_docs else "[No explicit context found]"

        logger.info(f"Retrieved {len(retrieved_docs)} documents from Pinecone")

        # 3. Build system prompt
        system_prompt = (
            "You are an intelligent assistant trained on the structure of this portfolio website. "
            "Answer the user query based purely on the website context provided below. "
            "CRITICAL: Be extremely concise, direct, and straightforward. Provide your answer in 1 to 3 sentences maximum. "
            "Avoid fluff, introductory remarks, or long bulleted lists. "
            "If the information is not present in the context, state exactly: 'I don't have that information on the website.'\n\n"
            f"Website Context:\n{context}"
        )

        # 4. Manage conversation history
        if session_id not in conversations_memory:
            conversations_memory[session_id] = []

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversations_memory[session_id])
        messages.append({"role": "user", "content": user_query})

        logger.info(f"Calling OpenRouter with key: {'SET' if OPENROUTER_KEY else 'MISSING'}")
        
        # 5. Call OpenRouter
        client_openai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
        response = client_openai.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=messages,
            max_tokens=150,
            temperature=0.3
        )
        ai_response = response.choices[0].message.content.strip()

        logger.info(f"Generated response: {ai_response}")

        # 6. Update memory
        conversations_memory[session_id].append({"role": "user", "content": user_query})
        conversations_memory[session_id].append({"role": "assistant", "content": ai_response})
        if len(conversations_memory[session_id]) > MAX_HISTORY_TURNS:
            conversations_memory[session_id] = conversations_memory[session_id][-MAX_HISTORY_TURNS:]

        return jsonify({"output": ai_response, "text": ai_response})

    except Exception as e:
        logger.exception("Chat endpoint error")
        return jsonify({"text": f"Internal error: {str(e)}"}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "awake"}), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
