# rag_pipeline.py
import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
from PyPDF2 import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth

# Load .env
load_dotenv()

# Config
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
INDEX_NAME = "rag-index"

# --- Clients (created lazily / safely) ---
# GROQ
GROQ_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None


# Pinecone (new SDK)
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_KEY:
    pc = Pinecone(api_key=PINECONE_KEY)
    # create index if missing
    names = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ required
        )
    index = pc.Index(INDEX_NAME)
else:
    pc = None
    index = None


# Neo4j (try primary, fallback if needed)
NEO_URI = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URI_PRIMARY") or os.getenv("NEO4J_URI_FALLBACK")
NEO_USER = os.getenv("NEO4J_USERNAME")
NEO_PWD = os.getenv("NEO4J_PASSWORD")

driver = None
if NEO_URI and NEO_USER and NEO_PWD:
    try:
        driver = GraphDatabase.driver(
            NEO_URI,
            auth=basic_auth(NEO_USER, NEO_PWD),
            encrypted=True  # ✅ required for Aura
        )
        # test connection immediately
        with driver.session(database="neo4j") as session:
            session.run("RETURN 1")
    except Exception as e:
        st.error(f"❌ Failed to connect to Neo4j: {e}")
# Embeddings
embedder = SentenceTransformer(EMBED_MODEL)


# --- Helpers ---
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def embed_and_store(chunks, doc_id):
    """Embed in batches, upsert to Pinecone (new SDK), store basic nodes in Neo4j (if configured)."""
    if index is None:
        raise RuntimeError("Pinecone not initialized (missing PINECONE_API_KEY).")

    BATCH = 200
    total = 0
    for b in range(0, len(chunks), BATCH):
        batch_chunks = chunks[b:b + BATCH]
        vecs = [embedder.encode(c).tolist() for c in batch_chunks]
        # upsert using new SDK Index.upsert with dict format
        items = [{"id": f"{doc_id}_chunk{b + i}", "values": vec, "metadata": {"text": batch_chunks[i]}} for i, vec in enumerate(vecs)]
        index.upsert(items)
        total += len(items)

    # insert into Neo4j if available
    if driver:
        with driver.session() as session:
            for i, chunk in enumerate(chunks):
                cid = f"{doc_id}_chunk{i}"
                session.run("CREATE (:Chunk {id:$id, text:$text})", id=cid, text=chunk)

    return total

def retrieve(query, top_k=3):
    if index is None:
        raise RuntimeError("Pinecone not initialized (missing PINECONE_API_KEY).")
    q_vec = embedder.encode(query).tolist()
    res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    # new SDK returns a dict-like response; matches under "matches"
    return res.get("matches", [])

def generate_answer(query, context_chunks):
    if client is None:
        return "⚠️ GROQ API key not configured — cannot generate answer."
    
    context_text = "\n".join([m["metadata"]["text"] for m in context_chunks])
    
    prompt = (
        "You are a knowledgeable assistant. Use the provided context to answer the question clearly and accurately. "
        "If the answer is not fully contained in the context, provide the best possible partial answer based on the context, "
        "and do not invent facts. Avoid saying 'I don’t know from the document' unless there is truly no relevant context.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}"
    )
    
    resp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2
    )
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return str(resp)


def visualize_neo4j_graph(limit=20):
    if driver is None:
        st.warning("Neo4j not configured, cannot visualize graph.")
        return
    with driver.session() as session:
        results = session.run("MATCH (c:Chunk) RETURN c.id AS id LIMIT $lim", lim=limit)
        ids = [r["id"] for r in results]
    if not ids:
        st.info("No nodes found in Neo4j.")
        return
    G = nx.Graph()
    for i in range(len(ids)-1):
        G.add_node(ids[i])
        G.add_edge(ids[i], ids[i+1])
    plt.figure(figsize=(6, 4))
    nx.draw(G, with_labels=True, node_size=400, font_size=8, node_color="lightblue")
    st.pyplot(plt)


# --- Streamlit UI ---
st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title("📄 RAG Pipeline with Pinecone + Neo4j + Groq")

# Sidebar env status (no secret values shown)
st.sidebar.header("🔐 Environment Status")
envs = {
    "GROQ_API_KEY": bool(GROQ_KEY),
    "PINECONE_API_KEY": bool(PINECONE_KEY),
    "NEO4J_URI": bool(NEO_URI),
    "NEO4J_USERNAME": bool(NEO_USER),
    "NEO4J_PASSWORD": bool(NEO_PWD),
}
for k, present in envs.items():
    if present:
        st.sidebar.success(f"{k} ✅")
    else:
        st.sidebar.error(f"{k} ❌")



# File upload
uploaded_files = st.file_uploader("📂 Upload PDF(s)", type="pdf", accept_multiple_files=True)
if uploaded_files:
    all_chunks = []
    total_pages = 0
    for f in uploaded_files:
        reader = PdfReader(f)
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join([t for t in pages if t.strip()])
        total_pages += len(reader.pages)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    st.success(f"✅ {len(uploaded_files)} document(s) uploaded — {total_pages} pages, {len(all_chunks)} chunks created.")
    st.write("Embeddings are not stored yet. Click below to store into Pinecone (and Neo4j if configured).")

    if st.button("Store Embeddings (Pinecone → Neo4j)"):
        if index is None:
            st.error("Pinecone not configured. Set PINECONE_API_KEY in environment or .env.")
        else:
            with st.spinner("Embedding & upserting..."):
                stored = embed_and_store(all_chunks, uploaded_files[0].name.replace(".", "_"))
            st.success(f"✅ Done — {stored} chunks embedded & stored (Neo4j updated if configured).")

# QA
st.subheader("🔎 Ask a Question")
q = st.text_input("Type your question here...")
k = st.slider("Top K chunks to retrieve", 1, 10, 5)
if q:
    if index is None:
        st.error("Pinecone not configured — cannot retrieve.")
    else:
        matches = retrieve(q, top_k=k)
        if not matches:
            st.warning("No relevant chunks found.")
        else:
            st.write("### 🔍 Retrieved chunks (top results):")
            for m in matches:
                st.info(m["metadata"]["text"][:600])
            answer = generate_answer(q, matches)
            st.success("📌 Answer:")
            st.write(answer)
