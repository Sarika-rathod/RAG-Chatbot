import os
import uuid
import logging
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
HF_TOKEN = os.getenv("HF_TOKEN")

# Override GROQ_MODEL in Render environment variables if needed.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

UPLOAD_DIR = Path("temp_uploads")
MAX_PDF_SIZE = 15 * 1024 * 1024  # 15 MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 16
MAX_HISTORY_MESSAGES = 12


app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = None
embeddings = None
pinecone_client = None
pinecone_index = None

session_store: Dict[str, BaseChatMessageHistory] = {}


def initialize_clients():
    """Initialize shared clients once per application process."""
    global llm, embeddings, pinecone_client, pinecone_index

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing.")

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing.")

    if llm is None:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0,
        )

    if embeddings is None:
        logger.info("Configuring hosted Hugging Face embeddings.")

        # Uses remote inference rather than loading PyTorch locally.
        embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            task="feature-extraction",
            huggingfacehub_api_token=HF_TOKEN,
        )

    if pinecone_index is None:
        if pinecone_client is None:
            pinecone_client = Pinecone(
                api_key=PINECONE_API_KEY,
            )

        existing_indexes = pinecone_client.list_indexes().names()

        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(
                "Creating Pinecone index: %s",
                PINECONE_INDEX_NAME,
            )

            pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

        pinecone_index = pinecone_client.Index(
            PINECONE_INDEX_NAME,
        )

        logger.info("Pinecone initialized.")



def get_session_history(
    session_id: str,
) -> BaseChatMessageHistory:
    """Return conversation history for a session."""

    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()

    return session_store[session_id]


def trim_session_history(session_id: str):
    """Keep recent messages to prevent unlimited RAM growth."""

    history = get_session_history(session_id)

    if len(history.messages) > MAX_HISTORY_MESSAGES:
        history.messages[:] = history.messages[
            -MAX_HISTORY_MESSAGES:
        ]



@app.get("/")
async def serve_html():
    """Serve the chatbot frontend."""

    html_path = Path("chatbot_ui.html")

    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="chatbot_ui.html was not found.",
        )

    return FileResponse(html_path)


@app.get("/health")
async def health():
    """Lightweight health check without initializing models."""

    return {
        "status": "ok",
        "vector_database": "Pinecone",
        "index": PINECONE_INDEX_NAME,
    }



@app.post("/load_pdf/")
async def load_pdf_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Process a PDF incrementally and store chunks in Pinecone."""

    if not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id cannot be empty.",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    file_location = None

    try:
        initialize_clients()

        UPLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Use a generated filename to avoid path traversal.
        file_location = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"

        total_size = 0

        # Read in small blocks rather than loading the whole upload
        # into RAM with await file.read().
        with file_location.open("wb") as output_file:
            while True:
                block = await file.read(1024 * 1024)

                if not block:
                    break

                total_size += len(block)

                if total_size > MAX_PDF_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 15 MB upload limit.",
                    )

                output_file.write(block)

        vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=session_id,
        )

        loader = PyPDFLoader(str(file_location))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        batch = []
        chunk_count = 0

        # Process one page at a time.
        for page in loader.lazy_load():
            page_chunks = splitter.split_documents([page])

            for document in page_chunks:
                document.metadata["session_id"] = session_id
                document.metadata["chunk_id"] = chunk_count
                document.metadata["source_file"] = Path(
                    file.filename
                ).name

                chunk_count += 1
                batch.append(document)

                if len(batch) >= BATCH_SIZE:
                    vectorstore.add_documents(batch)
                    batch.clear()

        # Upload the final partial batch.
        if batch:
            vectorstore.add_documents(batch)
            batch.clear()

        logger.info(
            "Processed PDF: %s chunks for session %s",
            chunk_count,
            session_id,
        )

        return {
            "message": "Uploaded successfully",
            "session_id": session_id,
            "chunks": chunk_count,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("PDF processing failed.")

        return JSONResponse(
            status_code=500,
            content={"error": f"{type(e)._name_}:{str(e)}" },
        )

    finally:
        await file.close()

        if file_location and file_location.exists():
            file_location.unlink(missing_ok=True)


 
class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=200)



@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """Answer questions using the session's Pinecone namespace."""

    prompt = request.prompt.strip()
    session_id = request.session_id.strip()

    if not prompt or not session_id:
        raise HTTPException(
            status_code=400,
            detail="prompt and session_id cannot be empty.",
        )

    try:
        initialize_clients()

        vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=session_id,
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        contextualize_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Given the chat history and the latest user question, "
                "rewrite the question as a standalone question. "
                "Do not answer it.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_prompt,
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Answer the user's question using the provided context. "
                "Be concise and clear. If the context does not contain "
                "the answer, say that you do not know.\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        document_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain,
        )

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = await conversational_chain.ainvoke(
            {"input": prompt},
            config={
                "configurable": {
                    "session_id": session_id,
                },
            },
        )

        trim_session_history(session_id)

        return {
            "answer": response["answer"],
        }

    except Exception:
        logger.exception("Chat request failed.")

        return JSONResponse(
            status_code=500,
            content={
                "error": (
                    "The chatbot could not process your question. "
                    "Check the Render logs for details."
                ),
            },
        )



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
