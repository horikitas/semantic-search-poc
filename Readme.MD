# semantic-search-poc

A minimal GenAI project that creates OpenAI-powered text embeddings and enables semantic search using Qdrant Cloud
— designed for clean, reproducible vector search without relying on full LLM chat.

```shell
python -m venv venv
source venv/bin/activate       # On Mac/Linux
venv\Scripts\activate   # On Windows

pip install -r requirements.txt

python main.py

```

### First time Run
`python main.py`

### Subsequently just for search
`python search.py`