from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello! The server is running!'

if __name__ == '__main__':
    print("Starting test server on port 8000...")
    # Try different host settings
    try:
        app.run(host='0.0.0.0', port=8000, debug=True)
    except Exception as e:
        print(f"Error: {e}")
