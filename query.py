import os
from contextlib import asynccontextmanager
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------
# Load Environment Variables from .env file
# ------------------------------------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hindi-doc-index")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure mandatory credentials are present
    if not PINECONE_API_KEY or not GEMINI_API_KEY:
        raise RuntimeError("Missing required API keys in environment/.env configuration.")

    # 1. Initialize Multilingual Embedding Model
    print("Loading embedding model...")
    resources["embed_model"] = SentenceTransformer("intfloat/multilingual-e5-small")

    # 2. Initialize Pinecone Client
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    resources["pinecone_index"] = pc.index(INDEX_NAME)

    # 3. Initialize Google GenAI Client
    print("Initializing Gemini client...")
    resources["gemini_client"] = genai.Client(api_key=GEMINI_API_KEY)

    yield
    resources.clear()

app = FastAPI(title="Hindi RAG Search API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allows requests from any origin/domain
    allow_credentials=False,    # MUST be False when using wildcard "*" origins
    allow_methods=["*"],        # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],        # Allows all request headers
)

# ------------------------------------------------------------------------------
# Pydantic Schemas for Structured Response
# ------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., example="What is the key information?")
    top_k: Optional[int] = Field(default=3, ge=1, le=10)

class ChunkResult(BaseModel):
    id: str
    score: float
    text: str
    source: Optional[str] = None
    chunk_index: Optional[int] = None

class ChatResponse(BaseModel):
    original_query: str
    search_query_hindi: str
    answer: str
    retrieved_chunks: List[ChunkResult]

# ------------------------------------------------------------------------------
# Helper Function for Query Translation
# ------------------------------------------------------------------------------
def translate_query_to_hindi(client: genai.Client, query: str) -> str:
    """Translates non-Hindi/English input to Hindi.

    If the query is already in Hindi, returns it unchanged.
    """
    translation_prompt = (
        "Translate the following user input into clear, natural Hindi if it is in English "
        "or any other language. If the input is already written in Hindi script (Devanagari), "
        "return it exact as it is without changing anything. "
        "Return ONLY the final Hindi text, with no explanations, quotes, or additional formatting."
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"User Query: {query}",
        config=types.GenerateContentConfig(
            system_instruction=translation_prompt,
            temperature=0.0
        )
    )
    return response.text.strip()

# ------------------------------------------------------------------------------
# RAG Endpoint
# ------------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_search(payload: ChatRequest):
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Query string cannot be empty."
        )

    try:
        embed_model = resources["embed_model"]
        pinecone_index = resources["pinecone_index"]
        gemini_client = resources["gemini_client"]

        # Step 1: Pre-process and Translate Query to Hindi if necessary
        hindi_search_query = translate_query_to_hindi(gemini_client, payload.query)

        # Step 2: Embed Hindi query string (with E5 model required prefix)
        formatted_query = f"query: {hindi_search_query}"
        query_vector = embed_model.encode(formatted_query).tolist()

        # Step 3: Retrieve top-k relevant chunks from Pinecone
        pinecone_response = pinecone_index.query(
            vector=query_vector,
            top_k=payload.top_k,
            include_metadata=True
        )

        matches = pinecone_response.get("matches", [])

        # Parse retrieved chunks into structured output models
        retrieved_chunks = [
            ChunkResult(
                id=match["id"],
                score=round(match["score"], 4),
                text=match.get("metadata", {}).get("text", ""),
                source=match.get("metadata", {}).get("source"),
                chunk_index=match.get("metadata", {}).get("chunk_index")
            )
            for match in matches
        ]

        if not matches:
            return ChatResponse(
                original_query=payload.query,
                search_query_hindi=hindi_search_query,
                answer="दस्तावेज़ में इस प्रश्न से संबंधित कोई जानकारी नहीं मिली।",
                retrieved_chunks=[]
            )

        # Step 4: Combine retrieved context text blocks
        context_text = "\n\n".join(
            [f"--- Context {i+1} ---\n{chunk.text}" for i, chunk in enumerate(retrieved_chunks)]
        )

        # Step 5: Construct RAG Prompt for Gemini
        system_instruction = (
            "your name is Premand Ji Maharaj and you are a radha Krishna bhakt , brahmachari, and a spiritual guide. "
            "you have to instruct the user, by answering their queries related to bhagwat, "
            "bhagwad gita, naam jap, bhakti, and other spiritual topics. Naam jap is very important. "
            "you have to answer in hindi language only.do not give any marks like *,।,1."
        )

        user_prompt = f"""
संदर्भ (Context):
{context_text}

प्रश्न (Question):
{hindi_search_query}

उत्तर (Hindi):
"""

        # Step 6: Generate final spiritual response using Gemini
        response = gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )

        return ChatResponse(
            original_query=payload.query,
            search_query_hindi=hindi_search_query,
            answer=response.text.strip(),
            retrieved_chunks=retrieved_chunks
        )

    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(err)}"
        )