import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lawchatbot.config import AppConfig
from lawchatbot.weaviate_client import initialize_weaviate_client
from lawchatbot.vectorstore import initialize_vector_store
from lawchatbot.retrievers import (
    initialize_semantic_retriever,
    initialize_bm25_retriever,
    initialize_hybrid_retriever,
    wrap_retriever_with_source,
)
from lawchatbot.rag_chain import initialize_llm, build_rag_chain

# Load variables from a local .env file when present (no-op in production
# environments where the variables are injected directly).
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

_init_lock = Lock()
_system = {}


def setup_gpu_optimization():
    """Configure GPU settings for optimal performance (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ No GPU detected, using CPU")


def _initialize_system():
    """Build the RAG pipeline once and cache it in the module-level _system dict."""
    with _init_lock:
        if _system:
            return

        weaviate_url = os.getenv("WEAVIATE_URL", "")
        weaviate_key = os.getenv("WEAVIATE_API_KEY", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")

        print(f"🔍 WEAVIATE_URL: {'SET' if weaviate_url else 'NOT_SET'}")
        print(f"🔍 WEAVIATE_API_KEY: {'SET' if weaviate_key else 'NOT_SET'}")
        print(f"🔍 OPENROUTER_API_KEY: {'SET' if openrouter_key else 'NOT_SET'}")

        if not (weaviate_url and weaviate_key and openrouter_key):
            print("❌ Error: Missing required environment variables!")
            print("Set WEAVIATE_URL, WEAVIATE_API_KEY and OPENROUTER_API_KEY "
                  "(see .env.example).")
            return

        config = AppConfig(
            weaviate_url=weaviate_url,
            weaviate_api_key=weaviate_key,
            openai_api_key=openrouter_key,
            weaviate_class=os.getenv("WEAVIATE_CLASS", "JustiaFederalCases"),
            text_key="text",
            metadata_attributes=["text"],
            semantic_k=10,
            bm25_k=10,
            alpha=0.5,
        )

        print("🔄 Initializing system components...")
        try:
            client = initialize_weaviate_client(config)
            vectorstore = initialize_vector_store(client, config)
            semantic_ret = initialize_semantic_retriever(vectorstore, config)
            bm25_ret = initialize_bm25_retriever(client, config)
            hybrid_ret = initialize_hybrid_retriever(semantic_ret, bm25_ret, alpha=config.alpha)
            hybrid_ret = wrap_retriever_with_source(hybrid_ret)
            llm = initialize_llm()
            rag_chain = build_rag_chain(llm, hybrid_ret)

            _system.update({"client": client, "rag_chain": rag_chain})

            print("⏳ Pre-warming system with a dummy query...")
            rag_chain.invoke({"question": "This is a warmup question."})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("✅ System pre-warmed and ready.")
        except Exception as e:
            print(f"❌ Initialization failed: {type(e).__name__}: {e}")
            return


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_gpu_optimization()
    _initialize_system()
    yield
    client = _system.get("client")
    if client:
        try:
            client.close()
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared on shutdown")


app = FastAPI(title="LawChatbot", lifespan=lifespan)

# CORS — restrict origins in production via the ALLOWED_ORIGINS env var
# (comma-separated). Defaults to "*" for convenience during evaluation.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def get_system():
    return _system


@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if get_system().get("rag_chain"):
        return {"status": "healthy", "message": "LawChatbot is ready"}
    return JSONResponse(
        {"status": "unhealthy", "message": "System not initialized"},
        status_code=503,
    )


@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()

    if not question:
        return JSONResponse({"answer": "Please enter a question.", "context": []}, status_code=400)

    rag_chain = get_system().get("rag_chain")
    if not rag_chain:
        return JSONResponse(
            {
                "answer": "❌ System not initialized. Check the logs and ensure "
                          "environment variables are set correctly.",
                "context": [],
            },
            status_code=503,
        )

    try:
        response = rag_chain.invoke({"question": question})
        answer = response.get("answer", "")
        documents = response.get("source_documents", [])
        context = [doc.page_content for doc in documents] if documents else []
        return JSONResponse({"answer": answer, "context": context})
    except Exception as e:
        return JSONResponse(
            {"answer": f"Error: {type(e).__name__}: {e}", "context": []},
            status_code=500,
        )
