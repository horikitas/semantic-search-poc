from embed import get_embedding
from qdrant import search_similar

query = input("Enter a semantic search query: ")
print(f"Getting embeddings for {query} from OpenAI")
query_vector = get_embedding(query)

print(f"Searching for {query} in Qdrant collection with query vector fetched from OpenAI.")
results = search_similar(query_vector)

print("\n Top Matches:")
for r in results:
    print(f">>> {r}")
