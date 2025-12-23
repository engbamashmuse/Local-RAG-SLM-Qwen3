# Local LLM RAG Application

🚀 **Document Intelligence - Fully Local & Private**

A production-ready Retrieval-Augmented Generation (RAG) application powered by Tier-Selectable LLMs that runs entirely on your local machine. Upload documents, index them, and chat with them using natural language - all without sending data to the cloud.

## 🎯 Features

### Core Functionality
- ✅ **Multiple Document Upload**: Upload PDF, DOCX, and TXT files simultaneously
- ✅ **Folder Support**: Upload entire folders as ZIP files for batch processing
- ✅ **Document Collections**: Organize documents into collections (e.g., safety_manual, operations)
- ✅ **Intelligent Q&A**: Ask questions about your documents using natural language
- ✅ **Conversation Memory**: Maintains chat history with sliding window memory (1500 tokens)
- ✅ **Source Citations**: Every answer includes source document references
- ✅ **Vector Search**: ChromaDB with similarity threshold for accurate retrieval
- ✅ **Real-time Chat**: Interactive chat interface with message history
- ✅ **Health Monitoring**: System status checks for Ollama connectivity

### Technical Features
- 🔒 **100% Local**: No data leaves your machine
- 🧠 **Smart Chunking**: Optimized 500-character chunks with 50-character overlap
- 📊 **Metadata Filtering**: Query specific collections or all documents
- 🔄 **Duplicate Detection**: Automatic file hash-based deduplication
- 💾 **Persistent Storage**: MongoDB for metadata, ChromaDB for vectors
- ⚡ **Optimized for 8GB RAM**: Configured for efficient operation on limited hardware

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **LLM**: Tier-Selectable LLMs (via Ollama)
- **Embeddings**: nomic-embed-text (via Ollama)
- **Vector Database**: ChromaDB
- **Database**: MongoDB
- **RAG Framework**: LangChain
- **Document Parsers**: PyPDF, python-docx, TextLoader

### Frontend
- **Framework**: React 19
- **UI Library**: Shadcn/UI + Radix UI
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **Notifications**: Sonner

## 📋 Prerequisites

### System Requirements
- **CPU**: 11th Gen i5 or better (tested on i5-1145G7)
- **RAM**: 8GB minimum (configured for optimal performance)
- **Disk**: 10GB free space (for models and documents)
- **OS**: Linux, macOS, or Windows with WSL2

### Software Requirements
1. **Ollama** (required for local LLM)
2. **MongoDB** (configured in backend/.env)
3. **Node.js** 18+ and Yarn
4. **Python** 3.11+

## 🚀 Installation & Setup

### Step 1: Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull required models (Example)
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

**Note**: Qwen 2.5 3B Q4 quantized uses ~2GB RAM, leaving 5-6GB for the OS and application.

### Step 2: Backend Setup

```bash
cd /app/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify .env configuration
cat .env
# Should contain:
# MONGO_URL="mongodb://localhost:27017"
# DB_NAME="rag_slm_db"
# CORS_ORIGINS="*"
```

### Step 3: Frontend Setup

```bash
cd /app/frontend

# Install dependencies
yarn install

# Verify .env configuration
cat .env
# Should contain:
# REACT_APP_BACKEND_URL=<your-backend-url>
```

### Step 4: Start Services

**Production (with Supervisor)**:
```bash
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
```

**Development**:
```bash
# Terminal 1 - Backend
cd /app/backend
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 - Frontend
cd /app/frontend
yarn start
```

## 📖 Usage Guide

### 1. Upload Documents

**Single/Multiple Files**:
1. Navigate to the "Documents" tab
2. Click "Select Files" and choose one or more files (PDF, DOCX, TXT)
3. Enter a collection name (e.g., "operations_manual")
4. Click "Upload & Index"

**Folder Upload**:
1. Compress your folder into a ZIP file
2. Click "Choose ZIP Folder"
3. Select the ZIP file
4. Enter a collection name
5. Click "Upload & Index"

**Supported Formats**:
- PDF (.pdf)
- Word Documents (.docx)
- Text Files (.txt)
- ZIP Folders (.zip) containing the above formats

### 2. Chat with Documents

1. Navigate to the "Chat" tab
2. (Optional) Select a specific collection from the dropdown
3. Type your question in natural language
4. Press Enter or click Send
5. View the AI's response with source citations

**Example Questions**:
- "What are the safety procedures for equipment maintenance?"
- "Summarize the key points from the operations manual"
- "What does the document say about emergency protocols?"

### 3. Manage Documents

- **View All Documents**: See all indexed documents in the Documents tab
- **Filter by Collection**: Use the collection filter in the chat interface
- **Delete Documents**: Click the trash icon next to any document
- **Clear Chat**: Click "Clear" to start a new conversation

## 🔧 Configuration

### RAM Optimization Settings

**Backend (server.py)**:
```python
# LLM Configuration
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.1,
    num_ctx=8192  # Context window optimized for 8GB RAM
)

# Memory Configuration
memory = ConversationTokenBufferMemory(
    max_token_limit=1500,  # Prevents RAM crashes
)

# Chunking Configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Optimized for CPU efficiency
    chunk_overlap=50
)
```

### Model Selection (New)

The system now supports a **3-Tier Model Architecture** managed via the UI.

1.  **Open the Web Interface** (`http://localhost:3000`).
2.  **System Configuration Panel**: Use the dropdowns to select a tier (Low/Mid/High) and a specific model.
    *   **Low**: < 4GB RAM (e.g., `phi3:mini`)
    *   **Mid**: 4-8GB RAM (e.g., `qwen2.5:3b`)
    *   **High**: > 16GB RAM (e.g., `llama3.1:70b`)
3.  **Set Active Model**: Click the button. The system will handle the switch asynchronously. You can watch the progress indicator.
    *   *Note: First-time switches may take minutes to download the model.*

**Manual Override**:
You can still manually update the source of truth if needed:
```bash
echo "qwen2.5:3b" > backend/.model
```

### Pre-set Prompts
Use the **"Sparkles" icon** in the chat bar to insert pre-made templates like:
*   **Smart Summary**: Generates a structured summary of the context.
*   **Action Items**: Extracts actions into a markdown table.

## 🛡️ Deployment & Safety

*   **Version Pinning**: All Docker images are pinned by SHA digest in `docker-compose.yml` to ensure reproducibility.
*   **CI Gate**: The `scripts/ci_gate.py` script ensures no hardcoded model names (e.g., "qwen2.5") are introduced into the codebase.
*   **Recovery**: The system includes a rollback test (`scripts/verify_rollback.py`) to ensure resilience against failed model updates.

## 🐛 Troubleshooting

### Ollama Not Running
**Symptoms**: Health check shows "Ollama disconnected"
**Solution**:
```bash
# Check if Ollama is running
pgrep ollama

# If not, start it
ollama serve &

# Verify models are installed
ollama list
```

### Out of Memory Errors
**Symptoms**: Server crashes or freezes during queries
**Solutions**:
1. Reduce `num_ctx` from 8192 to 4096
2. Lower `max_token_limit` from 1500 to 1000
3. Use smaller model like `qwen2.5:1.5b`

### Document Upload Fails
**Symptoms**: "Failed to upload document" error
**Common Causes**:
1. **File Format**: Ensure file is PDF, DOCX, or TXT
2. **Corrupted File**: Try opening the file in its native application
3. **Large Files**: For files >50 pages, split into smaller documents

### Chat Responses Are Slow
**Expected Performance**: 5-8 seconds per query on i5-1145G7
**Optimization Tips**:
1. Reduce `num_ctx` for faster but less context-aware responses
2. Increase `score_threshold` from 0.5 to 0.7 (fewer but more relevant chunks)
3. Close other resource-intensive applications

## 📊 Performance Benchmarks

**Hardware**: 11th Gen i5-1145G7 | 8GB RAM | No GPU

| Metric | Value |
|--------|-------|
| Model Load Time | ~5s |
| Document Indexing | ~2s per page |
| Query Response Time | 5-8s |
| Tokens/Second | 8-12 |
| RAM Usage (Idle) | ~3GB |
| RAM Usage (Query) | ~5GB |

## 🔐 Security & Privacy

- ✅ **No Cloud Dependencies**: All processing happens locally
- ✅ **No API Keys Required**: No external services needed
- ✅ **Data Sovereignty**: Your documents never leave your machine
- ✅ **File Hash Verification**: Automatic duplicate detection
- ✅ **Secure Storage**: Documents stored in isolated uploads directory

## 📝 API Documentation

### Health Check
```bash
GET /api/health
Response: {"status": "healthy", "ollama": "connected"}
```

### Upload Documents
```bash
POST /api/documents/upload
Content-Type: multipart/form-data

Body:
- files: File[] (multiple files)
- collection: string

Response: {
  "message": "Processed 3 file(s)",
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "details": [...]
}
```

### Chat
```bash
POST /api/chat
Content-Type: application/json

Body: {
  "query": "What are the safety protocols?",
  "collection": "safety_manual",  // optional
  "session_id": "session_123"     // optional
}

Response: {
  "answer": "...",
  "sources": ["safety_manual.pdf"],
  "session_id": "session_123"
}
```

### List Documents
```bash
GET /api/documents?collection=operations

Response: {
  "documents": [...],
  "count": 5
}
```

### Delete Document
```bash
DELETE /api/documents/{document_id}

Response: {"message": "Document deleted successfully"}
```

## 🚢 Deployment

### Local Deployment
The application is configured to run locally with Supervisor managing both services.

### Production Deployment Considerations
1. **Database**: Use dedicated MongoDB instance
2. **Storage**: Configure persistent volumes for uploads and chroma_db
3. **Ollama**: Ensure Ollama runs as a system service
4. **Monitoring**: Set up health check endpoints
5. **Backups**: Regular backups of MongoDB and document uploads

## 🤝 Contributing

This is a production-ready local RAG-SLM application. For issues or improvements, please ensure:
1. Ollama is properly configured
2. All dependencies are installed
3. MongoDB is running
4. Models are downloaded

## 📄 License

This project uses open-source models and libraries:
- Qwen 2.5 3B: Apache 2.0 License
- LangChain: MIT License
- FastAPI: MIT License
- React: MIT License

## 🙏 Acknowledgments

- **Ollama**: For making local LLM deployment simple
- **Qwen Team**: For the excellent Qwen models
- **LangChain**: For the RAG framework
- **ChromaDB**: For efficient vector storage

---

**Built with ❤️ for local-first AI applications**

For questions or support, check the troubleshooting section or review the health check endpoint.