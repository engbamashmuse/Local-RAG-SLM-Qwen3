import { useState, useEffect, useRef } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Upload, FileText, MessageSquare, Trash2, Send, Database, Loader2, FolderOpen, AlertCircle, CheckCircle2 } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Home = () => {
  const [documents, setDocuments] = useState([]);
  const [collections, setCollections] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [collection, setCollection] = useState("default");
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);
  
  // Chat state
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [selectedCollection, setSelectedCollection] = useState("all");
  const [healthStatus, setHealthStatus] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchDocuments();
    fetchCollections();
    checkHealth();
    // Generate session ID
    setSessionId(`session_${Date.now()}`);
  }, []);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API}/health`);
      setHealthStatus(response.data);
    } catch (error) {
      console.error("Health check failed:", error);
      setHealthStatus({ status: "unhealthy", ollama: "disconnected" });
    }
  };

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/documents`);
      setDocuments(response.data.documents || []);
    } catch (error) {
      console.error("Error fetching documents:", error);
      toast.error("Failed to fetch documents");
    } finally {
      setLoading(false);
    }
  };

  const fetchCollections = async () => {
    try {
      const response = await axios.get(`${API}/collections`);
      setCollections(response.data.collections || []);
    } catch (error) {
      console.error("Error fetching collections:", error);
    }
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      // Validate file types
      const allowedTypes = [".pdf", ".docx", ".txt", ".zip"];
      const invalidFiles = files.filter(file => {
        const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        return !allowedTypes.includes(fileExt);
      });
      
      if (invalidFiles.length > 0) {
        toast.error("Only PDF, DOCX, TXT, and ZIP files are supported");
        return;
      }
      setSelectedFiles(files);
    }
  };

  const handleFolderSelect = () => {
    folderInputRef.current?.click();
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      toast.error("Please select file(s) to upload");
      return;
    }

    try {
      setUploading(true);
      setUploadProgress({ current: 0, total: selectedFiles.length });
      
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append("files", file);
      });
      formData.append("collection", collection);

      const response = await axios.post(`${API}/documents/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      });

      const result = response.data;
      setUploadProgress(null);
      
      // Show detailed results
      if (result.successful > 0) {
        toast.success(`Successfully uploaded ${result.successful} file(s)`);
      }
      if (result.failed > 0) {
        toast.error(`Failed to upload ${result.failed} file(s)`);
      }
      
      // Show details for each file
      result.details.forEach(detail => {
        if (detail.status === "failed") {
          toast.error(`${detail.filename}: ${detail.reason}`);
        }
      });
      
      setSelectedFiles([]);
      setCollection("default");
      if (fileInputRef.current) fileInputRef.current.value = "";
      if (folderInputRef.current) folderInputRef.current.value = "";
      
      fetchDocuments();
      fetchCollections();
    } catch (error) {
      console.error("Error uploading documents:", error);
      toast.error(error.response?.data?.detail || "Failed to upload documents");
      setUploadProgress(null);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (documentId) => {
    try {
      await axios.delete(`${API}/documents/${documentId}`);
      toast.success("Document deleted successfully");
      fetchDocuments();
      fetchCollections();
    } catch (error) {
      console.error("Error deleting document:", error);
      toast.error("Failed to delete document");
    }
  };

  const handleSendMessage = async () => {
    if (!query.trim()) return;

    const userMessage = { role: "user", content: query };
    setMessages(prev => [...prev, userMessage]);
    setQuery("");
    setChatLoading(true);

    try {
      const response = await axios.post(`${API}/chat`, {
        query: query,
        collection: selectedCollection === "all" ? null : selectedCollection,
        session_id: sessionId
      });

      const assistantMessage = {
        role: "assistant",
        content: response.data.answer,
        sources: response.data.sources
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      toast.error(error.response?.data?.detail || "Failed to get response");
      // Remove user message if error occurred
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setChatLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setSessionId(`session_${Date.now()}`);
    toast.success("Chat cleared");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-teal-50">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl sm:text-5xl font-bold mb-3 bg-gradient-to-r from-slate-800 via-blue-700 to-teal-700 bg-clip-text text-transparent" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
            Local RAG-SLM
          </h1>
          <p className="text-base text-slate-600" style={{ fontFamily: 'Inter, sans-serif' }}>
            Document Intelligence with Qwen 3 8B • Fully Local & Private
          </p>
        </div>

        {/* Health Status Alert */}
        {healthStatus && healthStatus.ollama === "disconnected" && (
          <Alert className="mb-6 border-amber-500 bg-amber-50" data-testid="health-alert">
            <AlertCircle className="h-4 w-4 text-amber-600" />
            <AlertTitle className="text-amber-800">Ollama Not Running</AlertTitle>
            <AlertDescription className="text-amber-700">
              Please start Ollama with: <code className="bg-amber-100 px-2 py-1 rounded text-sm">ollama serve</code> and pull models: <code className="bg-amber-100 px-2 py-1 rounded text-sm">ollama pull qwen2.5:3b</code> and <code className="bg-amber-100 px-2 py-1 rounded text-sm">ollama pull nomic-embed-text</code>
            </AlertDescription>
          </Alert>
        )}

        {healthStatus && healthStatus.ollama === "connected" && (
          <Alert className="mb-6 border-green-500 bg-green-50" data-testid="health-success">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <AlertTitle className="text-green-800">System Ready</AlertTitle>
            <AlertDescription className="text-green-700">
              Ollama is connected and ready. You can upload documents and start chatting!
            </AlertDescription>
          </Alert>
        )}

        <Tabs defaultValue="chat" className="w-full">
          <TabsList className="grid w-full grid-cols-2 max-w-md mx-auto mb-6">
            <TabsTrigger value="chat" data-testid="chat-tab">
              <MessageSquare className="w-4 h-4 mr-2" />
              Chat
            </TabsTrigger>
            <TabsTrigger value="documents" data-testid="documents-tab">
              <Database className="w-4 h-4 mr-2" />
              Documents
            </TabsTrigger>
          </TabsList>

          {/* Chat Tab */}
          <TabsContent value="chat" className="space-y-4">
            <Card className="backdrop-blur-sm bg-white/80 border-slate-200">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Ask Questions</span>
                  {collections.length > 0 && (
                    <Select value={selectedCollection} onValueChange={setSelectedCollection}>
                      <SelectTrigger className="w-[180px]" data-testid="collection-filter">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Collections</SelectItem>
                        {collections.map(coll => (
                          <SelectItem key={coll} value={coll}>{coll}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </CardTitle>
                <CardDescription>
                  Query your documents using natural language
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px] w-full pr-4 mb-4" data-testid="chat-messages">
                  {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-slate-400">
                      <MessageSquare className="w-12 h-12 mb-3 opacity-50" />
                      <p className="text-sm">Upload documents and start asking questions</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {messages.map((message, idx) => (
                        <div
                          key={idx}
                          className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                          data-testid={`message-${message.role}-${idx}`}
                        >
                          <div
                            className={`max-w-[80%] rounded-lg px-4 py-3 ${
                              message.role === "user"
                                ? "bg-blue-600 text-white"
                                : "bg-slate-100 text-slate-900"
                            }`}
                          >
                            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                            {message.sources && message.sources.length > 0 && (
                              <div className="mt-2 pt-2 border-t border-slate-300">
                                <p className="text-xs font-semibold mb-1">Sources:</p>
                                <div className="flex flex-wrap gap-1">
                                  {message.sources.map((source, sidx) => (
                                    <Badge key={sidx} variant="secondary" className="text-xs">
                                      {source}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      {chatLoading && (
                        <div className="flex justify-start">
                          <div className="bg-slate-100 rounded-lg px-4 py-3">
                            <Loader2 className="w-5 h-5 animate-spin text-slate-600" />
                          </div>
                        </div>
                      )}
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </ScrollArea>

                <div className="flex gap-2">
                  <Input
                    placeholder="Ask a question about your documents..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage()}
                    disabled={chatLoading || documents.length === 0 || healthStatus?.ollama === "disconnected"}
                    data-testid="chat-input"
                  />
                  <Button
                    onClick={handleSendMessage}
                    disabled={chatLoading || !query.trim() || documents.length === 0 || healthStatus?.ollama === "disconnected"}
                    data-testid="send-button"
                  >
                    <Send className="w-4 h-4" />
                  </Button>
                  {messages.length > 0 && (
                    <Button
                      variant="outline"
                      onClick={handleClearChat}
                      data-testid="clear-chat-button"
                    >
                      Clear
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Documents Tab */}
          <TabsContent value="documents" className="space-y-4">
            {/* Upload Section */}
            <Card className="backdrop-blur-sm bg-white/80 border-slate-200">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Upload Documents
                </CardTitle>
                <CardDescription>
                  Upload multiple PDF, DOCX, TXT files or a ZIP folder
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="file-upload">Select Files</Label>
                    <Input
                      ref={fileInputRef}
                      id="file-upload"
                      type="file"
                      accept=".pdf,.docx,.txt,.zip"
                      multiple
                      onChange={handleFileChange}
                      disabled={uploading}
                      data-testid="file-upload-input"
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="folder-upload">Select Folder (as ZIP)</Label>
                    <div className="flex gap-2">
                      <Input
                        ref={folderInputRef}
                        id="folder-upload"
                        type="file"
                        accept=".zip"
                        onChange={handleFileChange}
                        disabled={uploading}
                        data-testid="folder-upload-input"
                        className="hidden"
                      />
                      <Button
                        variant="outline"
                        onClick={handleFolderSelect}
                        disabled={uploading}
                        className="w-full"
                        data-testid="folder-select-button"
                      >
                        <FolderOpen className="w-4 h-4 mr-2" />
                        Choose ZIP Folder
                      </Button>
                    </div>
                  </div>
                </div>

                {selectedFiles.length > 0 && (
                  <div className="bg-slate-50 p-3 rounded-lg" data-testid="selected-files-list">
                    <p className="text-sm font-medium mb-2">Selected Files ({selectedFiles.length}):</p>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                      {selectedFiles.map((file, idx) => (
                        <p key={idx} className="text-xs text-slate-600">
                          • {file.name} ({(file.size / 1024).toFixed(1)} KB)
                        </p>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  <Label htmlFor="collection">Collection</Label>
                  <Input
                    id="collection"
                    placeholder="default"
                    value={collection}
                    onChange={(e) => setCollection(e.target.value)}
                    disabled={uploading}
                    data-testid="collection-input"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Group documents by category (e.g., safety_manual, operations)
                  </p>
                </div>

                {uploadProgress && (
                  <div className="space-y-2">
                    <Progress value={(uploadProgress.current / uploadProgress.total) * 100} />
                    <p className="text-sm text-slate-600 text-center">
                      Processing {uploadProgress.current} of {uploadProgress.total} files...
                    </p>
                  </div>
                )}

                <Button
                  onClick={handleUpload}
                  disabled={selectedFiles.length === 0 || uploading || healthStatus?.ollama === "disconnected"}
                  className="w-full"
                  data-testid="upload-button"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Indexing Documents...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      Upload & Index ({selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''})
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Documents List */}
            <Card className="backdrop-blur-sm bg-white/80 border-slate-200">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  Indexed Documents ({documents.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex justify-center py-8">
                    <Loader2 className="w-8 h-8 animate-spin text-slate-600" />
                  </div>
                ) : documents.length === 0 ? (
                  <div className="text-center py-8 text-slate-500">
                    <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No documents uploaded yet</p>
                  </div>
                ) : (
                  <ScrollArea className="h-[400px] w-full pr-4">
                    <div className="space-y-3">
                      {documents.map((doc) => (
                        <div
                          key={doc.id}
                          className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors"
                          data-testid={`document-item-${doc.id}`}
                        >
                          <div className="flex-1">
                            <p className="font-medium text-sm text-slate-900">{doc.filename}</p>
                            <div className="flex items-center gap-2 mt-1">
                              <Badge variant="outline" className="text-xs">
                                {doc.collection}
                              </Badge>
                              <span className="text-xs text-slate-500">
                                {(doc.file_size / 1024).toFixed(1)} KB
                              </span>
                            </div>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleDeleteDocument(doc.id)}
                            data-testid={`delete-document-${doc.id}`}
                          >
                            <Trash2 className="w-4 h-4 text-red-600" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
      <Toaster position="top-right" />
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;