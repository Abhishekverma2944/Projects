from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA


class RAGPipeline:
    """
    A production-ready Retrieval-Augmented Generation (RAG) pipeline
    supporting multiple documents with FAISS vector store and HuggingFace models.
    """
    def __init__(self, 
                 data_paths,   # now accepts list of files or a folder
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 gen_model: str = "google/flan-t5-large",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 device: int = 0):
        """
        :param data_paths: A path or list of paths (files or folders) containing documents
        :param embedding_model: HuggingFace sentence transformer for embeddings
        :param gen_model: HuggingFace model for generation
        :param chunk_size: Max size of text chunks
        :param chunk_overlap: Overlap between chunks
        :param device: Device (0 = GPU, -1 = CPU)
        """
        if isinstance(data_paths, str) or isinstance(data_paths, Path):
            data_paths = [data_paths]  # single file
        self.data_paths = [Path(p) for p in data_paths]

        self.embedding_model = embedding_model
        self.gen_model = gen_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device

        # internal state
        self.documents = []   # will hold raw text of all docs
        self.chunks = None
        self.vector_store = None
        self.retriever = None
        self.qa = None

        self._load_data()
        self._prepare_chunks()
        self._build_vector_store()
        self._setup_pipeline()

    def _load_data(self):
        """Load all datasets into memory"""
        for path in self.data_paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
                self.documents.append(raw_text)
                print(f"Loaded {path.name} ({len(raw_text)} chars)")
        print(f"Total documents loaded: {len(self.documents)}")

    def _prepare_chunks(self):
        """Split text into chunks for embeddings"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        self.chunks = splitter.create_documents(self.documents)
        print(f"Total chunks: {len(self.chunks)}")

    def _build_vector_store(self):
        """Build FAISS vector store"""
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = FAISS.from_documents(self.chunks, embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

    def _setup_pipeline(self):
        """Setup LLM + RetrievalQA pipeline"""
        qa_pipeline = pipeline(
            "text2text-generation", 
            model=self.gen_model, 
            device=self.device
        )
        llm = HuggingFacePipeline(pipeline=qa_pipeline)
        self.qa = RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever)

    def search_docs(self, query: str):
        """Retrieve relevant docs"""
        retrieved_docs = self.retriever.invoke(query)
        print(f"Retrieved docs: {len(retrieved_docs)}")
        if retrieved_docs:
            print("Preview:\n", retrieved_docs[0].page_content[:500])
        return retrieved_docs

    def ask(self, query: str) -> str:
        """Ask a question using RAG"""
        result = self.qa.run(query)
        return result


if __name__ == "__main__":
    # Example: pass multiple files
    rag = RAGPipeline(data_paths=[
        "/content/alt.atheism.txt",
        "/content/comp.graphics.txt",
        "/content/content/comp.graphics.txt",
        "/content/comp.sys.ibm.pc.hardware.txt",
        "/content/comp.windows.x.txt",
        "/content/misc.forsale.txttxt"
    ])

    # Retrieval example
    rag.search_docs("What does this dataset say about religion?")

    # QA example
    answer = rag.ask("Which document talks about financial performance?")
    print("Answer:", answer)
