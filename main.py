import qdrant
from embed import get_embedding
from qdrant import init_collection, upload_texts, search_similar

# Sample dataset
sample_texts = [
    "Those who cannot acknowledge themselves will eventually fail. — Itachi Uchiha",
    "People live their lives bound by what they accept as correct and true.  — Itachi Uchiha",
    "Knowledge and awareness are vague, and perhaps better called illusions.  — Itachi Uchiha",
    "I always lied to you, told you to forgive me. But I was never able to forgive myself. — Itachi Uchiha" ,
    "No matter what you decide to do from now on, I will love you forever. — Itachi Uchiha",
    "You and I are flesh and blood. I’m always going to be there for you, even if it's only as an obstacle for you to overcome. — Itachi Uchiha" ,
    "Self-sacrifice… a nameless shinobi who protects peace within its shadow. — Itachi Uchiha" ,
    "Wake up to reality. Nothing ever goes as planned in this accursed world. — Madara Uchiha" ,
    "In this world, wherever there is light – there are also shadows. — Madara Uchiha— Madara Uchiha",
    "Power is not will, it is the phenomenon of physically making things happen.— Madara Uchiha",
    "When a man learns to love, he must bear the risk of hatred.— Madara Uchiha",
    "Hope is merely an illusion for the naive.— Madara Uchiha",
    "The longer you live… the more you realize that reality is just made of pain, suffering, and emptiness.— Madara Uchiha",
    "Man seeks peace, yet at the same time yearning for war… those are the two realms belonging solely to man.— Madara Uchiha",
    "I have long since closed my eyes… my only goal is in the darkness.— Sasuke Uchiha",
    "I'll protect the village from the shadows — a true Hokage in silence.— Sasuke Uchiha",
"A true shinobi is one who endures no matter what gets thrown at him… all the while making sure he protects what is precious to him.— Sasuke Uchiha"
]

# Step 0: Test connection
qdrant.test_qdrant_connection()

# Step 1: Init collection
init_collection()

# Step 2: Upload and embed sample data
upload_texts(sample_texts, get_embedding)
print("Uploaded sample texts to Qdrant.")

# Step 3: Accept query and search
query = input("Enter a semantic search query: ")
query_vector = get_embedding(query)
results = search_similar(query_vector)

print("\n Top Matches:")
for r in results:
    print(f">>> {r}")
