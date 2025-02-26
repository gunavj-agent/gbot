from flask import Flask, request, jsonify, render_template_string
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
import os
import tempfile
from typing import List

app = Flask(__name__)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
vector_store = None

# Try to load existing vector store or create a new one
if os.path.exists("data/vectorstore"):
    try:
        vector_store = FAISS.load_local("data/vectorstore", embeddings)
        print("Loaded existing document index")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        # Initialize an empty vector store
        vector_store = FAISS.from_texts(["Initial document"], embeddings)
        vector_store.save_local("data/vectorstore")
        print("Created new vector store")
else:
    # Create data directory if it doesn't exist
    os.makedirs("data/vectorstore", exist_ok=True)
    # Initialize an empty vector store
    vector_store = FAISS.from_texts(["Initial document"], embeddings)
    vector_store.save_local("data/vectorstore")
    print("Created new vector store")

# Global LLaMA model instance
llm = None

def init_llm():
    """Initialize the LLaMA model"""
    global llm
    if llm is None:
        print("Initializing new LLaMA model instance")
        llm = Llama(
            model_path="models/llama-2-7b-chat.gguf",
            n_ctx=1024,          # Increased context window
            n_threads=os.cpu_count(),  # Use all available CPU cores
            n_batch=32,          # Set batch size
            n_gpu_layers=32,     # Offload layers to GPU
            temperature=0.4,     # Adjust temperature for output randomness
            verbose=True,
            seed=42,             # Fixed seed for consistent responses
            f16_kv=True          # Use float16 for key/value cache
        )
    return llm

def process_document(file) -> List[str]:
    """Process uploaded document and return chunks"""
    filename = file.filename
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        file.save(tmp_file.name)
        file_path = tmp_file.name

    # Load document based on file type
    if filename.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif filename.endswith('.txt'):
        loader = TextLoader(file_path)
    elif filename.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        os.unlink(file_path)
        raise ValueError("Unsupported file type")

    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    os.unlink(file_path)
    return chunks

# HTML template for the chat interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Guna's Chatbot</title>
    <style>
        body { 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            font-family: Arial, sans-serif;
        }
        .chat-container {
            height: 400px;
            border: 1px solid #ccc;
            overflow-y: auto;
            padding: 20px;
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
        }
        .bot-message {
            background-color: #f5f5f5;
            margin-right: 20%;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex-grow: 1;
            padding: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .file-upload {
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>Guna's Chatbot</h1>
    
    <div class="file-upload">
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" id="file" name="file" accept=".pdf,.txt,.docx">
            <button type="submit">Upload Document</button>
        </form>
    </div>

    <div class="chat-container" id="chat-container"></div>
    
    <div class="input-container">
        <input type="text" id="user-input" placeholder="Type your message...">
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');

        function addMessage(message, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.textContent = message;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            addMessage(message, true);
            userInput.value = '';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message }),
                });
                const data = await response.json();
                addMessage(data.response, false);
            } catch (error) {
                console.error('Error:', error);
                addMessage('Sorry, there was an error processing your message.', false);
            }
        }

        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        document.getElementById('upload-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData();
            const fileInput = document.getElementById('file');
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                });
                const data = await response.json();
                addMessage(data.message, false);
            } catch (error) {
                console.error('Error:', error);
                addMessage('Sorry, there was an error uploading the document.', false);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'results': [], 'error': 'No query provided'}), 400

        query = data['query']
        k = data.get('k', 3)  # Number of results to return, default 3
        
        if vector_store is None:
            return jsonify({'results': [], 'error': 'No documents have been uploaded yet'}), 400
            
        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)
        
        # Format results
        formatted_results = [{
            'content': doc.page_content,
            'metadata': doc.metadata
        } for doc in results]
        
        return jsonify({
            'results': formatted_results,
            'message': f'Found {len(formatted_results)} relevant passages'
        })
    except Exception as e:
        return jsonify({'results': [], 'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400

    try:
        chunks = process_document(file)
        global vector_store
        
        if vector_store is None:
            vector_store = FAISS.from_documents(chunks, embeddings)
        else:
            vector_store.add_documents(chunks)
            
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        vector_store.save_local("data/vectorstore")
        return jsonify({'message': 'Document processed successfully!'})
    except Exception as e:
        return jsonify({'message': f'Error processing document: {str(e)}'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        query = data['message']
        k = data.get('k', 3)  # Number of relevant documents to retrieve
        
        # Initialize LLM if needed
        global llm
        if llm is None:
            llm = init_llm()
        
        # Get embeddings and search for relevant documents
        relevant_docs = []
        if vector_store is not None:
            try:
                # Get documents with scores
                relevant_docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
                # Split into docs and scores
                relevant_docs = [doc for doc, score in relevant_docs_with_scores]
                similarity_scores = [float(score) for doc, score in relevant_docs_with_scores]
                print(f"Found {len(relevant_docs)} relevant documents")
            except Exception as e:
                print(f"Error searching vector store: {e}")
                relevant_docs = []
                similarity_scores = []
        
        # Prepare the RAG prompt
        # Only use documents if they have high relevance
        relevant_docs_with_scores = list(zip(relevant_docs, similarity_scores))
        highly_relevant_docs = [doc for doc, score in relevant_docs_with_scores if score < 1.0]  # Lower score means more relevant
        
        if highly_relevant_docs:
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                  for i, doc in enumerate(highly_relevant_docs)])
        else:
            context = "No relevant documents found for this query."
        
        system_message = """
You are a helpful AI assistant. Follow these steps:
1. First, check if any relevant documents were found in the context.
2. If relevant documents were found:
   - Use them to provide a detailed answer
   - Cite which documents you used
3. If no relevant documents were found:
   - For mathematical questions: provide a clear, direct numerical answer
   - For other questions: provide a comprehensive answer based on your knowledge
   - Clearly state that you're not using any documents
"""

        prompt = f"""<|im_start|>system
You are a helpful AI assistant. If the provided context is relevant to the question, use it and cite the documents. If not, just answer directly from your knowledge.
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}
<|im_end|>
<|im_start|>assistant
"""

        # Generate response
        output = llm(prompt=prompt, 
                    max_tokens=500,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                    repeat_penalty=1.1)
        
        response = output['choices'][0]['text'].strip()
        
        # Clean up the response
        response = response.replace('<|im_end|>', '')
        response = response.replace('<|im_start|>', '')
        response = response.replace('assistant', '')
        response = response.strip()
        
        # Return both the response and the retrieved documents
        return jsonify({
            'response': response,
            'retrieved_documents': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score,
                    'relevance_score': f"{max(0, min(100, (1 - score/2) * 100)):.1f}%"  # Normalized to 0-100%
                } for doc, score in zip(relevant_docs, similarity_scores)
            ],
            'number_of_docs_found': len(relevant_docs)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500
        return jsonify({'response': f'Error generating response: {str(e)}'}), 500

# Vector Database Endpoints
@app.route('/search_vector', methods=['POST'])
def search_vector_route():
    if vector_store is None:
        return jsonify({"error": "Vector store not initialized."}), 500
    
    query = request.form['query']
    try:
        results = vector_store.similarity_search(query, k=5)
        return jsonify([{"content": doc.page_content, "metadata": doc.metadata} for doc in results])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/vector_data', methods=['GET'])
def vector_data_route():
    if vector_store is None:
        return jsonify({"error": "Vector store not initialized."}), 500
    
    try:
        # Get all documents
        all_vectors = vector_store.docstore._dict
        
        # Get embeddings for each document
        vector_data = []
        for doc_id, doc in all_vectors.items():
            vector_data.append({
                "doc_id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                # Note: We can't directly access individual vectors in FAISS
                # as it uses internal IDs that don't match our doc IDs
            })
        
        return jsonify({
            "total_vectors": len(vector_data),
            "status": "Vector store is initialized",
            "documents": vector_data,
            "vector_dimension": vector_store.index.d,  # Dimension of vectors
            "note": "Individual vector embeddings are not directly accessible due to FAISS internal structure"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_vector', methods=['POST'])
def add_vector_route():
    if vector_store is None:
        return jsonify({"error": "Vector store not initialized."}), 500
    
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    
    try:
        # Add the text to the vector store
        vector_store.add_texts([data['text']])
        return jsonify({"message": "Text added successfully."}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        # Initialize LLaMA model at startup
        print("Starting chatbot server...")
        print("Step 1: Initializing LLaMA model...")
        init_llm()
        print("Step 2: LLaMA model initialized successfully")
        
        # Run on port 8000
        print("Step 3: Starting web server on http://localhost:8000")
        # Allow connections from any host
        app.run(host='0.0.0.0', port=8000, debug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
