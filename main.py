from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv
import os
import uuid

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set")
if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY is not set")

app = FastAPI(title="RAG Chatbot API",
    description="Document-based RAG chatbot using LangChain, Pinecone and Groq",
    version="1.0.0"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = None

def get_llm():
    global llm
    if llm is None:
        print("Initializing Groq LLM...")
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="openai/gpt-oss-120b"
        )
        print("Groq LLM initialized.")
    return llm

session_store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str)->BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

embeddings = None
pc = None
pinecone_index = None
INDEX_NAME = PINECONE_INDEX_NAME
EMBEDDING_DIMENSION = 384
def initialize_services():
    global embeddings
    global pc
    global pinecone_index
    # Already initialized
    if embeddings is not None and pinecone_index is not None:
        return
    print("Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        }
    )
    print("Connecting to Pinecone...")
    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        print(
            f"Creating Pinecone index: {INDEX_NAME}"
        )
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Pinecone index created.")
    pinecone_index = pc.Index(
        INDEX_NAME
    )
    print(
        "Pinecone initialized successfully."
    )

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "RAG Chatbot API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "RAG chatbot is running",
        "vector_database": "Pinecone"
    }

@app.post("/load_pdf/")
async def load_pdf(
    file: UploadFile = File(...),
    session_id: str = "default"
):
    print(
        f"Loading PDF for session: {session_id}"
    )
    # Initialize heavy services only when needed
    initialize_services()
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        return {
            "status": "error",
            "message": "Only PDF files are supported."
        }
    # Create temporary file
    temp_filename = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        contents = await file.read()
        with open(
            temp_filename,
            "wb"
        ) as f:
            f.write(contents)
        
        loader = PyPDFLoader(
            temp_filename
        )
        documents = loader.load()
        if not documents:
            return {
                "status": "error",
                "message": "Could not extract content from PDF."
            }
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(
            documents
        )
        
        for index, chunk in enumerate(chunks):
            chunk.metadata["session_id"] = session_id
            chunk.metadata["chunk_id"] = index
            chunk.metadata["source_file"] = (
                file.filename
            )
       
        vector_store = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=session_id
        )
        vector_store.add_documents(
            chunks
        )
        return {
            "status": "success",
            "message": "PDF uploaded and indexed successfully.",
            "filename": file.filename,
            "session_id": session_id,
            "pages": len(documents),
            "chunks": len(chunks)
        }
    except Exception as e:
        print(
            f"PDF processing error: {str(e)}"
        )
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        # Remove temporary file
        if os.path.exists(temp_filename):
            os.remove(
                temp_filename
            )

@app.post("/chat")
async def chat(
    request: ChatRequest
):
    session_id = request.session_id
    question = request.question
    print(
        f"Chat request received for session: {session_id}"
    )
    # Initialize heavy services only when needed
    initialize_services()
    try:
       
        vector_store = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=session_id
        )
       
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3
            }
        )
        
        llm = get_llm()
       
        contextualize_q_system_prompt = """
Given a chat history and the latest user question,
rewrite the question so that it can be understood
without the chat history.
Do not answer the question.
Return only the rewritten standalone question.
"""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    contextualize_q_system_prompt
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
                llm,
                retriever,
                contextualize_q_prompt
            )
        )
       
        system_prompt = """
You are a helpful document-based AI assistant.
Answer the user's question using ONLY the
information available in the provided document context.
If the answer cannot be found in the document,
respond exactly:
"I don't know based on the uploaded document."
Do not make up information.
Keep answers clear, accurate and concise.
Context:
{context}
"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
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
        
        question_answer_chain = (
            create_stuff_documents_chain(
                llm,
                qa_prompt
            )
        )
        
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
       
        response = conversational_rag_chain.invoke(
            {
                "input": question
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )
        answer = response.get(
            "answer",
            ""
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "question": question,
            "answer": answer
        }
    except Exception as e:
        print(
            f"Chat error: {str(e)}"
        )
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                8000
            )
        ),
        reload=False
    )
