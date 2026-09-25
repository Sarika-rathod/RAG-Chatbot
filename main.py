from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Groq LLM --------------------
llm = None


def get_llm():
    global llm

    if llm is None:
        from langchain_groq import ChatGroq

        print("Initializing Groq LLM...")

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="openai/gpt-oss-120b",
        )

        print("Groq LLM initialized.")

    return llm


# -------------------- Chat History --------------------
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

session_store: Dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str):

    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()

    return session_store[session_id]


# -------------------- BM25 Retriever Store --------------------

retriever_store = {}



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


# -------------------- Upload PDF --------------------
@app.post("/load_pdf/")
async def load_pdf_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):

    print("Received PDF for processing")

    try:

        # Create temporary directory
        os.makedirs("temp_uploads", exist_ok=True)

        # Check file type
        if not file.filename.lower().endswith(".pdf"):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Only PDF files are allowed."
                }
            )

        # Secure filename
        safe_filename = os.path.basename(
            file.filename
        )

        file_location = os.path.join(
            "temp_uploads",
            safe_filename
        )

        # Read uploaded file
        file_content = await file.read()

        # Save PDF
        with open(file_location, "wb") as f:
            f.write(file_content)

        print(
            f"PDF saved: {file_location}"
        )

        # ----------------------------
        # Load PDF
        # ----------------------------

        from langchain_community.document_loaders import (
            PyPDFLoader
        )

        loader = PyPDFLoader(
            file_location
        )

        documents = loader.load()

        print(
            f"Loaded {len(documents)} pages"
        )

        # ----------------------------
        # Split text
        # ----------------------------

        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(
            documents
        )

        print(
            f"Created {len(splits)} chunks"
        )

        # ----------------------------
        # Create BM25 Retriever
        # ----------------------------

        from langchain_community.retrievers import (
            BM25Retriever
        )

        retriever = BM25Retriever.from_documents(
            splits,
            k=3
        )

        # Store retriever in memory
        retriever_store[session_id] = retriever

        print(
            "BM25 retriever created successfully"
        )

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


# -------------------- Chat Request --------------------
class ChatRequest(BaseModel):

    prompt: str
    session_id: str


# -------------------- Chat Endpoint --------------------
@app.post("/chat")
async def chat_with_pdf(
    request: ChatRequest
):

    prompt = request.prompt
    session_id = request.session_id

    print(
        "Received prompt for session:",
        session_id
    )

    try:

        # ----------------------------
        # Check retriever
        # ----------------------------

        if session_id not in retriever_store:

            return JSONResponse(
                status_code=400,
                content={
                    "error":
                    "Please upload a PDF first for this session."
                }
            )

        retriever = retriever_store[
            session_id
        ]

        # ----------------------------
        # Get LLM
        # ----------------------------

        language_model = get_llm()

        # ----------------------------
        # LangChain imports
        # ----------------------------

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

        # ----------------------------
        # Contextual question prompt
        # ----------------------------

        contextualize_q_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Given a chat history and the latest "
                        "user question which might reference "
                        "context, formulate a standalone "
                        "question."
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
        )

        # ----------------------------
        # History-aware retriever
        # ----------------------------

        history_aware_retriever = (
            create_history_aware_retriever(
                language_model,
                retriever,
                contextualize_q_prompt
            )
        )

        # ----------------------------
        # Answer prompt
        # ----------------------------

        qa_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful document assistant. "
                        "Answer the user's question using only "
                        "the provided document context. "
                        "Keep the answer concise and clear. "
                        "If the answer cannot be found in the "
                        "document context, say you don't know.\n\n"
                        "DOCUMENT CONTEXT:\n{context}"
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
        )

        # ----------------------------
        # Document chain
        # ----------------------------

        document_chain = (
            create_stuff_documents_chain(
                language_model,
                qa_prompt
            )
        )

        # ----------------------------
        # RAG chain
        # ----------------------------

        rag_chain = (
            create_retrieval_chain(
                history_aware_retriever,
                document_chain
            )
        )

        # ----------------------------
        # Conversation memory
        # ----------------------------

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

        # ----------------------------
        # Generate answer
        # ----------------------------

        response = (
            conversational_rag_chain.invoke(
                {
                    "input": prompt
                },

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
