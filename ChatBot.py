from flask import Flask, request, render_template, redirect
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.schema import HumanMessage
import requests
import os
from dotenv import load_dotenv

load_dotenv()


# -------------------------
# Custom Groq LLM Wrapper
# -------------------------
class GroqLLM(LLM):
    api_key: str
    model: str = "llama-3.1-8b-instant"
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        super().__init__(api_key=api_key, model=model)

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 512
        }
        res = requests.post(self.api_url, headers=headers, json=data)
        if res.status_code != 200:
            return f"⚠️ API Error {res.status_code}: {res.text}"
        return res.json()["choices"][0]["message"]["content"]


# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = GroqLLM(GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = None
pdf_uploaded = False


def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])


@app.route("/", methods=["GET", "POST"])
def index():
    global vectorstore, pdf_uploaded

    answer = ""
    if request.method == "POST":
        if "pdf_file" in request.files:
            pdf = request.files["pdf_file"]
            if pdf.filename.endswith(".pdf"):
                text = extract_text(pdf)

                # Split into chunks using LangChain
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.create_documents([text])

                # Create FAISS vectorstore
                vectorstore = FAISS.from_documents(docs, embeddings)
                pdf_uploaded = True
                return redirect("/")

        question = request.form.get("question", "")
        if question and pdf_uploaded:
            # Retrieve relevant chunks
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(question)
            context = "\n".join([d.page_content for d in docs])

            # Ask Groq LLM
            prompt = f"""You are a helpful assistant. Answer the question using the context below.

Context:
{context}

Question: {question}
Answer (use LaTeX for math if needed):"""
            answer = llm(prompt)

    return render_template("index.html", answer=answer, pdf_uploaded=pdf_uploaded)


if __name__ == "__main__":
    app.run(debug=True)
