from flask import Flask, request, jsonify, render_template_string
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

app = Flask(__name__)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
vector_store = None

# Load or create vector store
if os.path.exists("data/vectorstore"):
    vector_store = FAISS.load_local("data/vectorstore", embeddings)
else:
    os.makedirs("data/vectorstore", exist_ok=True)
    vector_store = FAISS.from_texts(["Initial document"], embeddings)
    vector_store.save_local("data/vectorstore")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Chatbot</title>
</head>
<body>
    <h1>Welcome to the Chatbot!</h1>
    <form action="/chat" method="post">
        <input type="text" name="message" placeholder="Type your message here..." required>
        <input type="submit" value="Send">
    </form>
    <div id="response"></div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    # Implement your chat logic here
    return jsonify({'response': 'Chat response'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
