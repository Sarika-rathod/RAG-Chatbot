from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv
import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
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

app = FastAPI()

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

session_store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = PINECONE_INDEX_NAME
EMBEDDING_DIMENSION = 384

existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {INDEX_NAME}")
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

pinecone_index = pc.Index(INDEX_NAME)

vectorstore_cache = {}

@app.get("/")
async def serve_html():
    return FileResponse("chatbot_ui.html")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "RAG chatbot is running",
        "vector_database": "Pinecone"
    }

@app.post("/load_pdf/")
async def load_pdf_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    print(f"Received PDF for session: {session_id}")

    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are allowed."}
        )

    try:
        os.makedirs("temp_uploads", exist_ok=True)

        safe_filename = os.path.basename(file.filename)
        file_location = os.path.join("temp_uploads", safe_filename)

        file_content = await file.read()

        with open(file_location, "wb") as f:
            f.write(file_content)

        print(f"PDF saved: {file_location}")

        loader = PyPDFLoader(file_location)
        documents = loader.load()

        print(f"Loaded {len(documents)} pages")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(documents)

        print(f"Created {len(splits)} chunks")

        for i, document in enumerate(splits):
            document.metadata["session_id"] = session_id
            document.metadata["chunk_id"] = i
            document.metadata["source_file"] = safe_filename

        print("Creating embeddings and uploading vectors to Pinecone...")

        vectorstore = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=session_id
        )

        vectorstore_cache[session_id] = vectorstore

        print("Successfully stored document vectors in Pinecone.")

        return {
            "message": "Uploaded successfully",
            "pages": len(documents),
            "chunks": len(splits),
            "vector_database": "Pinecone",
            "namespace": session_id
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

class ChatRequest(BaseModel):
    prompt: str
    session_id: str

@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    prompt = request.prompt
    session_id = request.session_id

    print("Received prompt for session:", session_id)

    try:
        if session_id not in vectorstore_cache:
            print("Loading Pinecone vector store...")

            vectorstore = PineconeVectorStore(
                index=pinecone_index,
                embedding=embeddings,
                namespace=session_id
            )

            vectorstore_cache[session_id] = vectorstore
        else:
            vectorstore = vectorstore_cache[session_id]

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        language_model = get_llm()

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Given a chat history and the latest user question which might reference context, formulate a standalone question."
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            language_model,
            retriever,
            contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful document assistant.

Answer the user's question using only the provided document context.

Keep the answer concise and clear.

If the answer cannot be found in the provided document context, say:
"I don't know based on the uploaded document."

DOCUMENT CONTEXT:
{context}"""
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(
            language_model,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda sid: get_session_history(sid),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        response = conversational_rag_chain.invoke(
            {"input": prompt},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        return {
            "answer": response["answer"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
