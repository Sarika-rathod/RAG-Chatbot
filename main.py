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
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

embeddings = None
pinecone_client = None
pinecone_index = None

EMBEDDING_DIMENSION = 384

session_store: Dict[str, BaseChatMessageHistory] = {}

def initialize_pinecone():
    global embeddings, pinecone_client, pinecone_index

    if embeddings is not None and pinecone_index is not None:
        return

    print("Initializing HuggingFace embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Connecting to Pinecone...")

    pinecone_client = Pinecone(
        api_key=PINECONE_API_KEY
    )

    existing_indexes = pinecone_client.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")

        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Pinecone index created.")

    pinecone_index = pinecone_client.Index(
        PINECONE_INDEX_NAME
    )

    print("Pinecone initialized successfully.")

def get_session_history(
    session_id: str
) -> BaseChatMessageHistory:

    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()

    return session_store[session_id]

@app.get("/")
async def serve_html():
    return FileResponse("chatbot_ui.html")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vector_database": "Pinecone",
        "index": PINECONE_INDEX_NAME
    }

@app.post("/load_pdf/")
async def load_pdf_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):

    print("Received PDF for processing")

    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Only PDF files are allowed."
            }
        )

    file_location = None

    try:
        initialize_pinecone()

        os.makedirs(
            "temp_uploads",
            exist_ok=True
        )

        file_location = (
            f"temp_uploads/{file.filename}"
        )

        with open(
            file_location,
            "wb"
        ) as f:
            f.write(await file.read())

        loader = PyPDFLoader(
            file_location
        )

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )

        splits = text_splitter.split_documents(
            documents
        )

        for i, document in enumerate(splits):
            document.metadata["session_id"] = session_id
            document.metadata["chunk_id"] = i
            document.metadata["source_file"] = file.filename

        vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=session_id
        )

        vectorstore.add_documents(
            splits
        )

        print("Successfully processed PDF")

        return {
            "message": "Uploaded successfully",
            "session_id": session_id,
            "chunks": len(splits)
        }

    except Exception as e:

        print(
            f"PDF processing error: {str(e)}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

    finally:

        if (
            file_location
            and os.path.exists(file_location)
        ):
            os.remove(file_location)

class ChatRequest(BaseModel):
    prompt: str
    session_id: str

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

        initialize_pinecone()

        vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=session_id
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3
            }
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Given a chat history and the latest user question which might reference context, formulate a standalone question."
            ),
            MessagesPlaceholder(
                "chat_history"
            ),
            (
                "human",
                "{input}"
            )
        ])

        history_aware_retriever = (
            create_history_aware_retriever(
                llm,
                retriever,
                contextualize_q_prompt
            )
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Use the context below to answer the question briefly and clearly. Limit your response to key information. If unsure, say you don't know.\n\n{context}"
            ),
            MessagesPlaceholder(
                "chat_history"
            ),
            (
                "human",
                "{input}"
            )
        ])

        document_chain = (
            create_stuff_documents_chain(
                llm,
                qa_prompt
            )
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )

        conversational_rag_chain = (
            RunnableWithMessageHistory(
                rag_chain,
                lambda sid: get_session_history(sid),
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
        )

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

        print(
            f"Chat error: {str(e)}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

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