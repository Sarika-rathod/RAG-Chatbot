# -------------------- Imports --------------------
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv
import os


# -------------------- Environment Setup --------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


# -------------------- FastAPI App Setup --------------------
app = FastAPI()


# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Lazy LLM Setup --------------------
llm = None
embeddings = None


def get_llm():
    """
    Create the Groq LLM only when it is actually needed.
    """
    global llm

    if llm is None:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="openai/gpt-oss-120b",
        )

    return llm


def get_embeddings():
    """
    Load Hugging Face embeddings only when needed.

    This prevents the embedding model from consuming RAM
    during FastAPI startup.
    """
    global embeddings

    if embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        print("Loading Hugging Face embedding model...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Embedding model loaded successfully.")

    return embeddings


# -------------------- In-Memory Stores --------------------
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

session_store: Dict[str, BaseChatMessageHistory] = {}
vectorstore_cache = {}


# -------------------- Session History --------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:

    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()

    return session_store[session_id]


# -------------------- Serve Frontend --------------------
@app.get("/")
async def serve_html():
    return FileResponse("chatbot_ui.html")


# -------------------- Health Check --------------------
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "RAG chatbot is running"
    }


# -------------------- Upload & Process PDF --------------------
@app.post("/load_pdf/")
async def load_pdf_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):

    print("Received PDF for processing")

    try:

        # Create directories
        os.makedirs("temp_uploads", exist_ok=True)
        os.makedirs("vectors", exist_ok=True)

        # Check file type
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Only PDF files are allowed."
                }
            )

        # Prevent unsafe filenames
        safe_filename = os.path.basename(file.filename)

        file_location = os.path.join(
            "temp_uploads",
            safe_filename
        )

        # Save PDF
        file_content = await file.read()

        with open(file_location, "wb") as f:
            f.write(file_content)

        print("PDF saved:", file_location)

        # Import only when required
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS

        # Load PDF
        loader = PyPDFLoader(file_location)

        documents = loader.load()

        print(
            f"PDF loaded successfully. Pages: {len(documents)}"
        )

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )

        splits = text_splitter.split_documents(documents)

        print(
            f"Created {len(splits)} text chunks"
        )

        # Load embeddings only now
        embedding_model = get_embeddings()

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(
            splits,
            embedding_model
        )

        # Save vector store
        vectorstore_path = os.path.join(
            "vectors",
            session_id
        )

        os.makedirs(
            vectorstore_path,
            exist_ok=True
        )

        vectorstore.save_local(
            vectorstore_path
        )

        # Cache vector store
        vectorstore_cache[session_id] = vectorstore

        print("Successfully processed PDF")

        return {
            "message": "Uploaded successfully",
            "pages": len(documents),
            "chunks": len(splits)
        }

    except Exception as e:

        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# -------------------- Chat Request Schema --------------------
class ChatRequest(BaseModel):
    prompt: str
    session_id: str


# -------------------- Chat Endpoint --------------------
@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):

    prompt = request.prompt
    session_id = request.session_id

    print(
        "Received prompt for session:",
        session_id
    )

    vectorstore_path = os.path.join(
        "vectors",
        session_id
    )

    try:

        # Import only when chat is requested
        from langchain_community.vectorstores import FAISS

        # Load embeddings only when required
        embedding_model = get_embeddings()

        # -----------------------------------------
        # Load vector store
        # -----------------------------------------

        if session_id not in vectorstore_cache:

            if not os.path.exists(vectorstore_path):

                return JSONResponse(
                    status_code=400,
                    content={
                        "error":
                        "Please load a PDF first for this session."
                    }
                )

            vectorstore = FAISS.load_local(
                vectorstore_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )

            vectorstore_cache[session_id] = vectorstore

        else:

            vectorstore = vectorstore_cache[session_id]

        # -----------------------------------------
        # Retriever
        # -----------------------------------------

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3
            }
        )

        # -----------------------------------------
        # LangChain imports
        # -----------------------------------------

        from langchain.chains import (
            create_history_aware_retriever,
            create_retrieval_chain
        )

        from langchain.chains.combine_documents import (
            create_stuff_documents_chain
        )

        from langchain_core.prompts import (
            ChatPromptTemplate,
            MessagesPlaceholder
        )

        from langchain_core.runnables.history import (
            RunnableWithMessageHistory
        )

        # -----------------------------------------
        # Get LLM
        # -----------------------------------------

        language_model = get_llm()

        # -----------------------------------------
        # Contextual question prompt
        # -----------------------------------------

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given a chat history and the latest user "
                    "question which might reference context, "
                    "formulate a standalone question."
                ),

                MessagesPlaceholder(
                    "chat_history"
                ),

                (
                    "human",
                    "{input}"
                )
            ]
        )

        history_aware_retriever = (
            create_history_aware_retriever(
                language_model,
                retriever,
                contextualize_q_prompt
            )
        )

        # -----------------------------------------
        # QA prompt
        # -----------------------------------------

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Use the context below to answer the "
                    "question briefly and clearly. "
                    "Limit your response to key information. "
                    "If the answer is not available in the "
                    "context, say you don't know.\n\n"
                    "{context}"
                ),

                MessagesPlaceholder(
                    "chat_history"
                ),

                (
                    "human",
                    "{input}"
                )
            ]
        )

        # -----------------------------------------
        # Document chain
        # -----------------------------------------

        document_chain = (
            create_stuff_documents_chain(
                language_model,
                qa_prompt
            )
        )

        # -----------------------------------------
        # RAG chain
        # -----------------------------------------

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )

        # -----------------------------------------
        # Conversation memory
        # -----------------------------------------

        conversational_rag_chain = (
            RunnableWithMessageHistory(
                rag_chain,
                lambda sid:
                    get_session_history(sid),

                input_messages_key="input",

                history_messages_key="chat_history",

                output_messages_key="answer"
            )
        )

        # -----------------------------------------
        # Generate answer
        # -----------------------------------------

        response = (
            conversational_rag_chain.invoke(
                {"input": prompt},

                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )
        )

        return {
            "answer": response["answer"]
        }

    except Exception as e:

        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )