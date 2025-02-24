import requests
import json

def test_vector_db():
    base_url = 'http://127.0.0.1:5000'
    
    # Test search endpoint
    def search_vectors(query):
        print(f"\nSearching for: {query}")
        response = requests.post(f'{base_url}/search_vector', data={'query': query})
        if response.status_code == 200:
            print("Search Results:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    # Test get all vectors endpoint
    def get_all_vectors():
        print("\nGetting all vectors:")
        response = requests.get(f'{base_url}/vector_data')
        if response.status_code == 200:
            print("All Vectors:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    # Test adding a vector
    def add_vector(text):
        print(f"\nAdding text: {text}")
        response = requests.post(f'{base_url}/add_vector', 
                               json={'text': text})
        if response.status_code == 201:
            print("Text added successfully!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    # Run tests
    try:
        # Test search
        search_vectors("test query")
        
        # Test get all vectors
        get_all_vectors()
        
        # Test adding a text to the vector store
        example_text = "This is a test document to add to the vector store."
        add_vector(example_text)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the Flask app is running.")

if __name__ == "__main__":
    test_vector_db()
