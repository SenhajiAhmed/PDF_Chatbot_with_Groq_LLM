from flask import Flask, request, render_template, redirect
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

GROQ_API_KEY = "YOUR_API_KEY"  # 🔁 Replace this with your Groq API key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])



def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]



def create_index(chunks, model):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings



def retrieve_context(question, chunks, embeddings, model, k=3):
    q_emb = model.encode([question])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    distances, indices = index.search(np.array(q_emb).astype("float32"), k)
    return "\n".join([chunks[i] for i in indices[0]])



def ask_groq(context, question, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""You are a helpful assistant. Answer the user's question based only on the following context.

Context:
{context}

Question: {question}
Answer (use LaTeX for any math):"""

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "top_p": 0.9
    }

    try:
        res = requests.post(GROQ_API_URL, headers=headers, json=data)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"



uploaded_chunks = []
uploaded_embeddings = []
pdf_uploaded = False



@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_chunks, uploaded_embeddings, pdf_uploaded

    answer = ""
    if request.method == "POST":
        if "pdf_file" in request.files:
            pdf = request.files["pdf_file"]
            if pdf.filename.endswith(".pdf"):
                text = extract_text(pdf)
                chunks = chunk_text(text)
                index, embeddings = create_index(chunks, model)
                uploaded_chunks = chunks
                uploaded_embeddings = embeddings
                pdf_uploaded = True
                return redirect("/")

        question = request.form.get("question", "")
        if question and pdf_uploaded:
            context = retrieve_context(question, uploaded_chunks, uploaded_embeddings, model)
            answer = ask_groq(context, question, GROQ_API_KEY)

    return render_template("index.html", answer=answer, pdf_uploaded=pdf_uploaded)

if __name__ == "__main__":
    app.run(debug=True)
