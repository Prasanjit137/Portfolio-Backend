---
title: Portfolio Backend
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Portfolio Chatbot Backend

This is the backend API for my portfolio chatbot. 

## Features
- Flask API
- Local SentenceTransformer embedding model (`all-MiniLM-L6-v2`)
- Pinecone vector database integration
- OpenRouter LLM integration

## Endpoints
- `POST /chat` - Accepts `{"chatInput": "string", "sessionId": "string"}` and returns the AI response.