RAGPipeline – Retrieval-Augmented Generation with LangChain & HuggingFace
📌 Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline in Python.
It allows you to:

Load and process multiple documents.

Split documents into manageable chunks.

Create semantic embeddings with HuggingFace models.

Store and query embeddings using a FAISS vector store.

Retrieve the most relevant chunks for a query.

Use a HuggingFace text generation model to answer questions based on the retrieved context.

This setup is useful for building chatbots, QA systems, or knowledge assistants that can reason over large collections of text.

⚙️ Features

Supports multiple text documents (list of paths).

Document chunking with configurable size & overlap.

Embedding generation using HuggingFace sentence transformers.

Vector similarity search with FAISS.

Integration with HuggingFace LLMs for question answering.

Modular OOP design (easy to extend for new retrievers/models).

🏗️ Project Structure
RAGPipeline/
│── rag_pipeline.py      # Main pipeline implementation
│── README.md            # Documentation (this file)
│── requirements.txt     # Dependencies
│── data/                # Folder for your text documents

📦 Installation

Clone this repo (or copy the rag_pipeline.py file).

Install dependencies:

pip install langchain langchain-community faiss-cpu sentence-transformers transformers


Use faiss-gpu if you want GPU acceleration.

🚀 Usage
1. Import & Initialize
from rag_pipeline import RAGPipeline

rag = RAGPipeline(data_paths=[
    "data/alt.atheism.txt",
    "data/comp.graphics.txt"
])

2. Search for Relevant Documents
docs = rag.search_docs("What does this dataset say about religion?")

3. Ask Questions with RAG
answer = rag.ask("Which document talks about financial performance?")
print("Answer:", answer)

⚙️ Parameters

When initializing RAGPipeline, you can configure:

Parameter	Type	Default	Description
data_paths	str / list	Required	One or more paths to .txt documents.
embedding_model	str	sentence-transformers/all-MiniLM-L6-v2	HuggingFace model for embeddings.
gen_model	str	google/flan-t5-large	HuggingFace model for text generation.
chunk_size	int	1000	Size of text chunks for embeddings.
chunk_overlap	int	200	Overlap between chunks.
device	int	0	Device (0 = GPU, -1 = CPU).
🔍 Example Workflow

Documents are loaded into memory.

Text is split into overlapping chunks.

Each chunk is embedded into a vector.

FAISS builds a vector index.

Queries are matched against the index to retrieve top relevant chunks.

The HuggingFace LLM generates an answer using the retrieved context.

📚 Example Output
Loaded alt.atheism.txt (20394 chars)
Loaded comp.graphics.txt (17821 chars)
Total documents loaded: 2
Total chunks: 45
Retrieved docs: 4
Preview:
   [first 500 characters of the most relevant chunk...]
Answer: The document about financial performance is found in comp.graphics.txt.

🔮 Future Improvements

Support PDF, Word, and HTML file ingestion (via langchain.document_loaders).

Add persistence (save and reload FAISS index).

Swap HuggingFace LLM with OpenAI or Llama.



MIT License – free to use and modify.

👉 Do you want me to also include a requirements.txt file and example usage script so someone can just clone and run it immediately?
