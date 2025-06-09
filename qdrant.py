from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "semantic_search_demo"
VECTOR_DIM = 1536  # For text-embedding-3-small

QDRANT_HOST=os.getenv("QDRANT_HOST")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")


client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
)

def test_qdrant_connection():
    print("Calling qdrant client to test connection with host {}".format(QDRANT_HOST))
    print(client.get_collections())
    print("Completed qdrant connection")

def init_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def upload_texts(texts, embed_fn):
    points = [
        PointStruct(id=i, vector=embed_fn(t), payload={"text": t})
        for i, t in enumerate(texts)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search_similar(text_vector, k=3):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=text_vector,
        limit=k,
    )
    return [r.payload["text"] for r in results]
