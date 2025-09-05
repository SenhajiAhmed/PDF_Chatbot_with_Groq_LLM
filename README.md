## PDF Chatbot with Groq LLM

A simple Flask app that lets you upload a PDF, creates embeddings, retrieves relevant chunks, and answers questions using Groq LLM.

### Features
- PDF upload and text extraction
- Vector search with FAISS
- Embeddings via `all-MiniLM-L6-v2`
- Groq LLM (`llama-3.1-8b-instant`) for answers

### Requirements
- Python 3.13 (tested)
- Groq API key

### Setup
1. Clone the repo and enter the folder:
   ```bash
   git clone <your-repo-url>
   cd ChatBot
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Set your Groq API key securely (recommended using .env):
   - Create a `.env` file in the project root:
     ```bash
     echo "GROQ_API_KEY=YOUR_GROQ_API_KEY" > .env
     ```
   - `.gitignore` includes `.env` by default.

### Run
There are two implementations. Use one at a time.

1) LangChain version (`ChatBot.py`):
```bash
.venv/bin/python ChatBot.py
```
Then open `http://127.0.0.1:5000`.

2) Minimal version (previously `CB.py`):
- This has been merged into the LangChain version; prefer `ChatBot.py`.

### How it works (LangChain)
- Splits PDF with `RecursiveCharacterTextSplitter`
- Builds a FAISS vectorstore from chunks using `HuggingFaceEmbeddings`
- Retrieves top-k chunks per question
- Sends a composed prompt to Groq Chat Completions API

### Configuration
- Model name: `llama-3.1-8b-instant` (in code)
- Change embedding model by updating `HuggingFaceEmbeddings` in `ChatBot.py`

### Notes
- Large wheels (Torch/FAISS) can take time to install.
- For CPU-only PyTorch you can install from the CPU index if needed:
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
  ```




