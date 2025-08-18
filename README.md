
---

# 📄 Document Uploader with Pinecone + Neo4j + Embeddings

This project lets you upload PDF documents, split them into smaller chunks, create embeddings, and store them in **Pinecone** (for vector search) and **Neo4j** (for knowledge graph structure).

---

## 🚀 Features

* Upload PDF documents
* Extract text from PDFs
* Split text into chunks using `RecursiveCharacterTextSplitter`
* Generate embeddings with **all-MiniLM-L6-v2** (SentenceTransformers)
* Store embeddings in **Pinecone** (for similarity search)
* Store metadata and relationships in **Neo4j** (graph database)
* Returns total number of chunks stored

---



### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables (`.env`)

Create a `.env` file with your API keys:

```ini
PINECONE_API_KEY=your_pinecone_api_key
NEO4J_URI_PRIMARY=neo4j+s://<your-neo4j-uri>
NEO4J_URI_FALLBACK=neo4j+ssc://<your-neo4j-uri>
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### 4. Run App

```bash
python main.py
```

---

## 🔧 Tech Stack

* **Python**
* **LangChain** → for text splitting
* **SentenceTransformers** → `all-MiniLM-L6-v2` embeddings
* **Pinecone** → vector database
* **Neo4j** → knowledge graph database

---

##  Next Steps

* Add **retrieval + RAG pipeline** to answer questions
* Build a simple **frontend** (Streamlit or React) for interaction

---

