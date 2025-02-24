import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
import os
from typing import List
import tempfile
from flask import Flask, request, jsonify

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize FAISS vector store
if os.path.exists("data/vectorstore"):
    vector_store = FAISS.load_local("data/vectorstore", embeddings)
else:
    vector_store = None

def init_llm():
    """Initialize the LLaMA model"""
    return Llama(
        model_path="models/llama-2-7b-chat.gguf",  # Update with your model path
        n_ctx=2048,
        n_threads=4
    )

def process_document(file) -> List[str]:
    """Process uploaded document and return chunks"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        file_path = tmp_file.name

    # Load document based on file type
    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file.name.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file.name.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Apply embeddings to each chunk
    embeddings_list = []
    for chunk in chunks:
        embedding = embeddings.embed(chunk)  # Generate embedding for the chunk
        embeddings_list.append(embedding)
        # Optionally, store the embedding in the vector store
        if vector_store is not None:
            vector_store.add(embedding)  # Assuming vector_store has an 'add' method

    os.unlink(file_path)
    return chunks, embeddings_list

def search_vector(query):
    query_vector = embeddings.embed(query)  # Convert the query to a vector
    results = vector_store.search(query_vector, k=5)  # Search the vector store for the top 5 results
    return results

def main():
    st.title("Guna's Chatbot")
    
    # Document upload section
    st.sidebar.header("Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a document", type=['pdf', 'txt', 'docx'])
    
    if uploaded_file and st.sidebar.button("Process Document"):
        with st.spinner("Processing document..."):
            chunks, embeddings_list = process_document(uploaded_file)
            global vector_store
            
            if vector_store is None:
                vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                vector_store.add_documents(chunks)
                
            vector_store.save_local("data/vectorstore")
            st.sidebar.success("Document processed successfully!")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if vector_store is not None:
            # Retrieve relevant context
            docs = vector_store.similarity_search(prompt, k=3)
            context = "\n".join(doc.page_content for doc in docs)
            
            # Generate response using LLaMA
            llm = init_llm()
            system_prompt = f"You are a helpful assistant. Use the following context to answer the question: {context}"
            response = llm(f"{system_prompt}\n\nQuestion: {prompt}\n\nAnswer:", max_tokens=500)
            
            with st.chat_message("assistant"):
                st.markdown(response['choices'][0]['text'])
                st.session_state.messages.append({"role": "assistant", "content": response['choices'][0]['text']})
        else:
            with st.chat_message("assistant"):
                message = "Please upload and process some documents first!"
                st.markdown(message)
                st.session_state.messages.append({"role": "assistant", "content": message})

    # Search interface
    st.sidebar.header("Search Vector Database")
    search_query = st.sidebar.text_input("Search query")
    if st.sidebar.button("Search"):
        results = search_vector(search_query)
        st.write(results)

if __name__ == "__main__":
    app = Flask(__name__)

    # Define the Search Endpoint
    @app.route('/search_vector', methods=['POST'])
    def search_vector_route():
        query = request.form['query']  # Get the search query from the form
        results = search_vector(query)  # Call the search function
        return jsonify(results)  # Return the results as JSON

    # Define an Endpoint to Retrieve All Data
    @app.route('/vector_data', methods=['GET'])
    def vector_data_route():
        if vector_store is not None:
            all_vectors = vector_store.get_all_vectors()  # Replace with actual method to retrieve vectors
            return jsonify(all_vectors)
        else:
            return jsonify({"error": "Vector store not initialized."}), 500

    # Define an Endpoint to Add New Data
    @app.route('/add_vector', methods=['POST'])
    def add_vector_route():
        data = request.json  # Expecting JSON data
        vector = data.get('vector')  # Retrieve the vector from the request
        if vector_store is not None:
            vector_store.add(vector)  # Add the vector to the store
            return jsonify({"message": "Vector added successfully."}), 201
        else:
            return jsonify({"error": "Vector store not initialized."}), 500

    app.run()
