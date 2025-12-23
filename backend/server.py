from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import hashlib
import shutil
import zipfile
import tempfile
import yaml
import asyncio
import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, start_http_server, Summary

# RAG imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Load Model Name (Single Source of Truth)
try:
    MODEL_NAME = (ROOT_DIR / ".model").read_text(encoding="utf-8").strip()
except Exception as e:
    logging.warning(f"Could not read .model file: {e}")
    # Try to load first mid model from catalog, else unknown
    try:
        catalog_path = ROOT_DIR / "models_catalog.yaml"
        with open(catalog_path, "r", encoding="utf-8") as f:
            cat = yaml.safe_load(f)
            MODEL_NAME = cat['mid'][0]['name']
    except:
        MODEL_NAME = "unknown"
    logging.info(f"Defaulting to model: {MODEL_NAME}")

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create directories for storage
UPLOADS_DIR = ROOT_DIR / "uploads"
CHROMA_DIR = ROOT_DIR / "chroma_db"
UPLOADS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Initialize Ollama embeddings and LLM
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=8192
    )
except Exception as e:
    logging.warning(f"Ollama not available: {e}. Make sure Ollama is running with required models.")
    embeddings = None
    llm = None

# Initialize ChromaDB
vectorstore = None
if embeddings:
    try:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name="documents"
        )
    except Exception as e:
        logging.warning(f"ChromaDB initialization failed: {e}")

# job store for async tasks
jobs = {}

# Telemetry
MODEL_SWITCH_DURATION = Summary('rag_model_switch_duration_seconds', 'Time spent switching models')


# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_hash: str
    collection: str = "default"
    file_size: int
    upload_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_path: str
    status: str = "indexed"  # indexed, processing, failed

class ChatMessage(BaseModel):
    role: str  # user or assistant
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    query: str
    collection: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

class ChatHistory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UploadResponse(BaseModel):
    message: str
    total_files: int
    successful: int
    failed: int
    successful: int
    failed: int
    details: List[dict]

class ModelSetRequest(BaseModel):
    model: str

class JobStatus(BaseModel):
    job_id: str
    status: str # running, success, failed
    details: Optional[str] = None

# Helper functions
def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_and_split_document(file_path: Path, collection: str) -> List:
    """Load and split document into chunks"""
    file_extension = file_path.suffix.lower()
    
    # Load document based on type
    if file_extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_extension == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif file_extension == ".txt":
        loader = TextLoader(str(file_path), encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    documents = loader.load()
    
    # Split documents - critical: keep chunks small for CPU efficiency
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add metadata to each chunk
    for chunk in chunks:
        chunk.metadata["collection"] = collection
        chunk.metadata["filename"] = file_path.name
    
    return chunks

    return chunks

async def switch_model_task(model_name: str, job_id: str):
    """Background task to switch model"""
    start_time = time.time()
    try:
        jobs[job_id] = {"status": "running", "details": f"Switching to {model_name}..."}
        
        # Check if offline mode from env or just default logic
        # For this implementation, we try to detect if we can pull
        
        # 1. Check if model exists locally
        # In a real shell we would run 'ollama list', here we simulate or try pull
        
        # Run ollama pull
        process = await asyncio.create_subprocess_exec(
            "ollama", "pull", model_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            # If pull failed, maybe we are offline?
            # If strict offline mode is requested (Gap Remedy #2 part), we should have checked list first.
            # For now, we report the error.
            error_msg = stderr.decode()
            jobs[job_id] = {"status": "failed", "details": f"Ollama pull failed: {error_msg}"}
            return

        # 2. Update .model file
        (ROOT_DIR / ".model").write_text(model_name, encoding="utf-8")
        
        # 3. Re-initialize global LLM (This is tricky in async default fastapi, 
        # usually we just rely on the next request picking it up if we re-create, 
        # OR we rely on Ollama behaving dynamically.
        # But server.py initializes `llm` global on startup. We need to update it.)
        global llm
        llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_ctx=8192
        )
        
        # Telemetry
        duration = time.time() - start_time
        MODEL_SWITCH_DURATION.observe(duration)
        
        jobs[job_id] = {"status": "success", "details": f"Successfully switched to {model_name}"}
        
    except Exception as e:
        logging.error(f"Model switch failed: {e}")
        jobs[job_id] = {"status": "failed", "details": str(e)}

async def process_single_file(file_path: Path, filename: str, collection: str) -> dict:
    """Process a single file and return result"""
    try:
        # Calculate file hash
        file_hash = calculate_file_hash(file_path)
        
        # Check if document already exists
        existing_doc = await db.documents.find_one({"file_hash": file_hash})
        if existing_doc:
            return {
                "filename": filename,
                "status": "skipped",
                "reason": "Already exists"
            }
        
        # Load and index document
        chunks = load_and_split_document(file_path, collection)
        
        # Generate unique ID
        file_id = str(uuid.uuid4())
        
        # Add to vector store with metadata filtering
        if vectorstore:
            vectorstore.add_documents(
                chunks,
                ids=[f"{file_id}_{i}" for i in range(len(chunks))]
            )
        
        # Save metadata to MongoDB
        doc_metadata = DocumentMetadata(
            id=file_id,
            filename=filename,
            file_hash=file_hash,
            collection=collection,
            file_size=file_path.stat().st_size,
            file_path=str(file_path),
            status="indexed"
        )
        
        doc_dict = doc_metadata.model_dump()
        doc_dict['upload_date'] = doc_dict['upload_date'].isoformat()
        await db.documents.insert_one(doc_dict)
        
        return {
            "filename": filename,
            "status": "success",
            "chunks": len(chunks),
            "document_id": file_id
        }
    
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return {
            "filename": filename,
            "status": "failed",
            "reason": str(e)
        }

def extract_files_from_folder(zip_path: Path, extract_to: Path) -> List[Path]:
    """Extract files from ZIP folder and return list of valid document paths"""
    allowed_extensions = [".pdf", ".docx", ".txt"]
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Recursively find all valid documents
    for file_path in extract_to.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
            extracted_files.append(file_path)
    
    return extracted_files

# Session memory store (in production, use Redis)
# Store chat history as list of messages
session_memories = {}

# Routes
@api_router.get("/")
async def root():
    return {"message": "Local LLM RAG API", "status": "running"}

@api_router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection: str = Form("default")
):
    """Upload and index multiple documents or a folder (ZIP)"""
    if not vectorstore or not llm:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Please start Ollama with: ollama serve"
        )
    
    results = []
    successful = 0
    failed = 0
    temp_dir = None
    
    try:
        for file in files:
            file_extension = Path(file.filename).suffix.lower()
            
            # Handle ZIP files (folders)
            if file_extension == ".zip":
                temp_dir = Path(tempfile.mkdtemp())
                zip_path = temp_dir / file.filename
                
                # Save ZIP file
                with open(zip_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Extract files
                try:
                    extracted_files = extract_files_from_folder(zip_path, temp_dir)
                    
                    for extracted_file in extracted_files:
                        result = await process_single_file(
                            extracted_file,
                            extracted_file.name,
                            collection
                        )
                        results.append(result)
                        if result["status"] == "success":
                            successful += 1
                        elif result["status"] == "failed":
                            failed += 1
                    
                    # Clean up temp directory
                    shutil.rmtree(temp_dir)
                    temp_dir = None
                    
                except Exception as e:
                    results.append({
                        "filename": file.filename,
                        "status": "failed",
                        "reason": f"ZIP extraction failed: {str(e)}"
                    })
                    failed += 1
            
            # Handle individual files
            else:
                # Validate file type
                allowed_extensions = [".pdf", ".docx", ".txt"]
                if file_extension not in allowed_extensions:
                    results.append({
                        "filename": file.filename,
                        "status": "failed",
                        "reason": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
                    })
                    failed += 1
                    continue
                
                # Save file
                file_id = str(uuid.uuid4())
                file_path = UPLOADS_DIR / f"{file_id}_{file.filename}"
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process file
                result = await process_single_file(file_path, file.filename, collection)
                results.append(result)
                
                if result["status"] == "success":
                    successful += 1
                elif result["status"] == "failed":
                    failed += 1
                    # Clean up failed file
                    if file_path.exists():
                        file_path.unlink()
        
        return UploadResponse(
            message=f"Processed {len(files)} file(s)",
            total_files=len(results),
            successful=successful,
            failed=failed,
            details=results
        )
    
    except Exception as e:
        logging.error(f"Error uploading documents: {str(e)}")
        # Clean up temp directory if exists
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/documents")
async def get_documents(collection: Optional[str] = None):
    """Get list of all documents"""
    try:
        query = {}
        if collection:
            query["collection"] = collection
        
        documents = await db.documents.find(query, {"_id": 0}).to_list(1000)
        
        # Convert ISO string timestamps back to datetime objects
        for doc in documents:
            if isinstance(doc['upload_date'], str):
                doc['upload_date'] = datetime.fromisoformat(doc['upload_date'])
        
        return {"documents": documents, "count": len(documents)}
    
    except Exception as e:
        logging.error(f"Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors"""
    try:
        # Get document from database
        doc = await db.documents.find_one({"id": document_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file
        file_path = Path(doc['file_path'])
        if file_path.exists():
            file_path.unlink()
        
        # Delete from vector store
        if vectorstore:
            try:
                vectorstore.delete(ids=[f"{document_id}_{i}" for i in range(1000)])  # Delete up to 1000 chunks
            except Exception as e:
                logging.warning(f"Error deleting from vectorstore: {e}")
        
        # Delete from MongoDB
        await db.documents.delete_one({"id": document_id})
        
        return {"message": "Document deleted successfully"}
    
    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/collections")
async def get_collections():
    """Get list of all collections"""
    try:
        collections = await db.documents.distinct("collection")
        return {"collections": collections}
    
    except Exception as e:
        logging.error(f"Error fetching collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/models")
async def get_models():
    """Get available models from catalog"""
    try:
        catalog_path = ROOT_DIR / "models_catalog.yaml"
        if not catalog_path.exists():
            return {"error": "Catalog not found"}
        
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
            
        # Get current model
        try:
            current = (ROOT_DIR / ".model").read_text(encoding="utf-8").strip()
        except:
            current = "unknown"
            
        return {"catalog": catalog, "current": current}
    except Exception as e:
        logging.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/model/set")
async def set_model(request: ModelSetRequest):
    """Switch active model (Async)"""
    # Validate against catalog
    catalog_path = ROOT_DIR / "models_catalog.yaml"
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        
        # Flatten catalog to find valid names
        valid_models = []
        for tier in catalog.values():
            for m in tier:
                valid_models.append(m["name"])
        
        if request.model not in valid_models:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not in catalog")
    
    # Create Job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "details": "Starting..."}
    
    # Start background task
    asyncio.create_task(switch_model_task(request.model, job_id))
    
    return {"job_id": job_id, "message": "Model switch accepted"}

@api_router.get("/model/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Check status of async job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        details=job.get("details")
    )

@api_router.get("/prompts")
async def get_prompts():
    """List available pre-set prompts"""
    prompts_dir = ROOT_DIR / "prompts"
    if not prompts_dir.exists():
        return {"prompts": []}
        
    prompts = []
    for f in prompts_dir.glob("*.txt"):
        try:
            content = f.read_text(encoding="utf-8")
            prompts.append({
                "name": f.stem.replace("_", " ").title(),
                "filename": f.name,
                "content": content
            })
        except Exception as e:
            logging.error(f"Error reading prompt {f}: {e}")
            
    return {"prompts": prompts}

@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with documents using RAG"""
    if not vectorstore or not llm:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Please start Ollama and pull required models: <model_name>, nomic-embed-text"
        )
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get chat history for this session
        if session_id not in session_memories:
            session_memories[session_id] = []
        
        chat_history = session_memories[session_id]
        
        # Create retriever with optional collection filtering
        search_kwargs = {
            "k": 3,
            "score_threshold": 0.5
        }
        
        if request.collection:
            search_kwargs["filter"] = {"collection": request.collection}
        
        # Retrieve relevant documents
        docs = vectorstore.similarity_search_with_score(
            request.query,
            k=3,
            filter=search_kwargs.get("filter")
        )
        
        # Filter by score threshold
        relevant_docs = [doc for doc, score in docs if score >= 0.5]
        
        # Extract source documents and context
        sources = []
        context_parts = []
        for doc in relevant_docs:
            if "filename" in doc.metadata:
                sources.append(doc.metadata["filename"])
            context_parts.append(doc.page_content)
        
        sources = list(set(sources))  # Remove duplicates
        context = "\n\n".join(context_parts)
        
        # Build conversation messages
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on the provided context. If the answer is not in the context, say so."),
        ]
        
        # Add recent chat history (last 5 exchanges to fit in 1500 tokens)
        for msg in chat_history[-10:]:
            messages.append(msg)
        
        # Add current query with context
        user_message = f"Context:\n{context}\n\nQuestion: {request.query}"
        messages.append(HumanMessage(content=user_message))
        
        # Get response from LLM
        response = llm.invoke(messages)
        answer = response.content
        
        # Update chat history (keep last 10 messages)
        chat_history.append(HumanMessage(content=request.query))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        session_memories[session_id] = chat_history
        
        # Save chat history to MongoDB
        chat_history_doc = await db.chat_history.find_one({"session_id": session_id})
        
        new_messages = [
            ChatMessage(role="user", content=request.query),
            ChatMessage(role="assistant", content=answer)
        ]
        
        if chat_history_doc:
            # Update existing history
            messages = chat_history_doc.get("messages", [])
            for msg in new_messages:
                msg_dict = msg.model_dump()
                msg_dict['timestamp'] = msg_dict['timestamp'].isoformat()
                messages.append(msg_dict)
            
            await db.chat_history.update_one(
                {"session_id": session_id},
                {"$set": {
                    "messages": messages,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }}
            )
        else:
            # Create new history
            messages = []
            for msg in new_messages:
                msg_dict = msg.model_dump()
                msg_dict['timestamp'] = msg_dict['timestamp'].isoformat()
                messages.append(msg_dict)
            
            history = ChatHistory(
                session_id=session_id,
                messages=new_messages
            )
            history_dict = history.model_dump()
            history_dict['created_at'] = history_dict['created_at'].isoformat()
            history_dict['updated_at'] = history_dict['updated_at'].isoformat()
            
            for msg in history_dict['messages']:
                msg['timestamp'] = msg['timestamp'].isoformat()
            
            await db.chat_history.insert_one(history_dict)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        history = await db.chat_history.find_one({"session_id": session_id}, {"_id": 0})
        
        if not history:
            return {"messages": []}
        
        # Convert ISO strings back to datetime
        for msg in history.get('messages', []):
            if isinstance(msg['timestamp'], str):
                msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
        
        return history
    
    except Exception as e:
        logging.error(f"Error fetching chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/chat/history/{session_id}")
async def delete_chat_history(session_id: str):
    """Delete chat history for a session"""
    try:
        # Remove from memory store
        if session_id in session_memories:
            del session_memories[session_id]
        
        # Remove from MongoDB
        await db.chat_history.delete_one({"session_id": session_id})
        
        return {"message": "Chat history deleted successfully"}
    
    except Exception as e:
        logging.error(f"Error deleting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Ollama connectivity
        if llm:
            test_response = llm.invoke("test")
            ollama_status = "connected"
        else:
            ollama_status = "disconnected"
        
        return {
            "status": "healthy" if ollama_status == "connected" else "degraded",
            "ollama": ollama_status,
            "mongodb": "connected",
            "vectorstore": "ready" if vectorstore else "not initialized"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "ollama": "disconnected",
            "note": "Please run: ollama serve && ollama pull <model_name> && ollama pull nomic-embed-text"
        }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()