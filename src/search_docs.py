import requests
import json
from typing import List, Dict

def search_documents(query: str, k: int = 3) -> List[Dict]:
    """
    Search through uploaded documents using FAISS.
    
    Args:
        query: The search query
        k: Number of results to return (default: 3)
        
    Returns:
        List of relevant document chunks with their content and metadata
    """
    url = "http://localhost:8080/search"
    data = {
        "query": query,
        "k": k
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise exception for bad status codes
        
        results = response.json()
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return []
            
        print(f"\nFound {len(results['results'])} relevant passages:")
        for i, result in enumerate(results['results'], 1):
            print(f"\nPassage {i}:")
            print(f"Content: {result['content']}")
            print(f"Metadata: {result['metadata']}")
            
        return results['results']
        
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

if __name__ == "__main__":
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        k = input("How many results do you want? (default: 3): ")
        try:
            k = int(k) if k else 3
        except ValueError:
            k = 3
            
        results = search_documents(query, k)
