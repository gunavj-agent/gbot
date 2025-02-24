# Guna's RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot that combines the power of LLaMA with document context. Built using Flask, FAISS for vector storage, LangChain for document processing, and HuggingFace for embeddings.

## Features

- **RAG Implementation**: Enhances LLaMA responses with relevant document context
- **Smart Document Retrieval**: Uses FAISS for efficient similarity search
- **Relevance Scoring**: Shows document relevance scores for transparency
- **Adaptive Responses**: 
  - Uses document context when relevant
  - Falls back to model knowledge for general questions
  - Handles mathematical queries directly
- **Multiple Document Types**: Supports PDF, TXT, and DOCX files

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download LLaMA model:
- Download LLaMA 2 7B chat model (GGUF format)
- Place in `models` directory

4. Start the Flask server:
```bash
export FLASK_ENV=development
flask run
```

## API Endpoints

### Document Management
- `POST /upload`: Upload and process documents
- `GET /vector_data`: View stored document data

### Chat Interface
- `POST /chat`: Send queries and get RAG-enhanced responses
  ```json
  {
    "message": "your question here",
    "k": 3  // optional: number of relevant docs to retrieve
  }
  ```

### Vector Operations
- `POST /search_vector`: Search for similar vectors
- `POST /add_vector`: Add new vectors to the store

## Response Format
```json
{
  "response": "Generated answer",
  "retrieved_documents": [
    {
      "content": "Document content",
      "metadata": {},
      "relevance_score": "85.5%"
    }
  ],
  "number_of_docs_found": 1
}
```

## Implementation Details

1. **Document Processing**:
   - Documents are split into chunks
   - Chunks are converted to embeddings
   - Embeddings are stored in FAISS

2. **Query Processing**:
   - User query is converted to embedding
   - Similar documents are retrieved
   - Relevance scores are calculated

3. **Response Generation**:
   - Context is prepared from relevant documents
   - LLaMA generates response using context
   - Response includes relevance metrics

## Contributing

Feel free to submit issues and enhancement requests!
