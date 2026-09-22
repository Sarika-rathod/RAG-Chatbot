# -------------------- Imports --------------------
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv
import os

# LangChain imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------- Environment Setup --------------------
# Load environment variables from .env file
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# -------------------- FastAPI App Setup --------------------
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Global LLM & Embedding Setup --------------------
# Instantiate Groq LLM with LLaMA model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b"
)

# Use HuggingFace miniLM embeddings for vector similarity
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# -------------------- In-Memory Stores --------------------
# For managing session history and vectorstores
session_store: Dict[str, BaseChatMessageHistory] = {}
vectorstore_cache: Dict[str, FAISS] = {}

# -------------------- Session History Handler --------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# -------------------- Serve Frontend --------------------
@app.get("/")
async def serve_html():
    return FileResponse("chatbot_ui.html")

# -------------------- Upload & Process PDF --------------------
@app.post("/load_pdf/")
async def load_pdf_upload(file: UploadFile = File(...), session_id: str = Form(...)):
    print("Received PDF for processing")
    os.makedirs("temp_uploads", exist_ok=True)
    os.makedirs("vectors", exist_ok=True)

    # Check file type
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are allowed."})

    # Save uploaded PDF temporarily
    file_location = f"temp_uploads/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        # Load and split PDF into text chunks
        loader = PyPDFLoader(file_location)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Convert to vectorstore and save to disk
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore_path = f"vectors/{session_id}"
        os.makedirs(vectorstore_path, exist_ok=True)
        vectorstore.save_local(vectorstore_path)

        # Cache vectorstore for quick access
        vectorstore_cache[session_id] = vectorstore
        print("Successfully processed PDF")
        return {"message": "Uploaded successfully"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------- Chat Request Schema --------------------
class ChatRequest(BaseModel):
    prompt: str
    session_id: str

# -------------------- Chat Endpoint --------------------
@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    prompt = request.prompt
    session_id = request.session_id
    print("Received prompt for session:", session_id)

    vectorstore_path = f"vectors/{session_id}"

    # If vectorstore not cached, attempt to load from disk
    if session_id not in vectorstore_cache:
        if os.path.exists(vectorstore_path):
            try:
                vectorstore = FAISS.load_local(
                    vectorstore_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                vectorstore_cache[session_id] = vectorstore
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Please load a PDF first for this session."}
            )

    retriever = vectorstore_cache[session_id].as_retriever()

    # Create a prompt to rephrase context-dependent queries
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context, formulate a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Final answering prompt using context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Use the context below to answer the question **briefly and clearly in short**. "
         "Limit your response to key information. If unsure, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Combine retrieval with conversation memory
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda sid: get_session_history(sid),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Generate response
    response = conversational_rag_chain.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": session_id}},
    )

    return {"answer": response["answer"]}