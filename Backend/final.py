import os
import warnings
import logging
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Disable warnings and reduce logging noise
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration paths
DOCUMENTS_FOLDER = "it"
PERSIST_DIRECTORY = "faiss_index"
FUTURE_DOCS_FOLDER = r"C:\Users\dell\Desktop\itc\future_docs"
FUTURE_PERSIST_DIRECTORY = "future_faiss_index"

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

def print_progress(current, total, stage):
    progress = (current / total) * 100
    print(f"\r[{stage}] Progress: {current}/{total} ({progress:.1f}%)", end="")
    if current == total:
        print()

def load_documents_from_folder(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        raise ValueError(f"Documents folder not found: {folder_path}")

    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    total_files = len(file_list)
    print(f"\nFound {total_files} PDF files in {folder_path}")

    for i, file_name in enumerate(file_list, 1):
        file_path = os.path.join(folder_path, file_name)
        try:
            print(f"\nProcessing file {i}/{total_files}: {file_name}")
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()

            for page_num, page in enumerate(pages, 1):
                page.metadata["source"] = file_name
                page.metadata["page"] = page_num

            documents.extend(pages)
            print(f"Processed {len(pages)} pages.")
            print_progress(i, total_files, "LOADING")
        except Exception as e:
            print(f"\nError loading {file_name}: {str(e)}")
            continue
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents

def get_vectorstore(folder_path, persist_directory):
    try:
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
        if os.path.exists(persist_directory):
            print(f"\nLoading FAISS index from {persist_directory}")
            vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
            return vectorstore, None

        print(f"\nCreating FAISS index for {folder_path}")
        documents = load_documents_from_folder(folder_path)
        if not documents:
            return None, f"No valid documents found in {folder_path}"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        split_documents = text_splitter.split_documents(documents)
        print(f"Created {len(split_documents)} chunks")

        vectorstore = FAISS.from_documents(split_documents, embeddings)
        vectorstore.save_local(persist_directory)
        return vectorstore, None

    except Exception as e:
        return None, f"Error processing documents from {folder_path}: {str(e)}"

# Initialize vectorstores
print("Initializing document vectorstores...")
current_vectorstore, current_error = get_vectorstore(DOCUMENTS_FOLDER, PERSIST_DIRECTORY)
future_vectorstore, future_error = get_vectorstore(FUTURE_DOCS_FOLDER, FUTURE_PERSIST_DIRECTORY)
print("Initialization complete.")

def get_chatgroq_model(temperature=0.1):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set in the environment.")
    return ChatGroq(model_name="llama3-8b-8192", temperature=temperature)

@app.post("/api/policy-answer")
async def get_policy_answer(request: QuestionRequest):
    try:
        if current_error:
            return {"answer": current_error}
        if current_vectorstore is None:
            return {"answer": "Current tax documents not available."}

        llm = get_chatgroq_model(temperature=0.1)
        template = """You are an expert on Indian Income Tax laws and regulations. Answer the question based only on the following context:
        {context}

        Question: {question}

        Provide a concise answer with relevant sections from the Income Tax Act 1961. 
        DO NOT mention any document names or page references.
        Only provide the legal information in clear, simple language.
        If you don't know, say "I couldn't find this information in the Income Tax Act"."""
        prompt_template = ChatPromptTemplate.from_template(template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=current_vectorstore.as_retriever(search_kwargs={'k': 6}),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )

        result = qa_chain({"query": request.question})
        response = result["result"].replace("According to Income Tax Act", "The Income Tax Act states")
        return {"answer": response}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

@app.post("/api/web-answer")
async def get_web_answer(request: QuestionRequest):
    try:
        llm = get_chatgroq_model(temperature=0.5)

        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
        if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
            raise EnvironmentError("Google API keys not configured.")

        def google_custom_search(query, num=3):
            url = "https://www.googleapis.com/customsearch/v1"
            params = {'q': query, 'key': GOOGLE_API_KEY, 'cx': SEARCH_ENGINE_ID, 'num': num}
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                return response.json().get('items', [])
            except Exception as e:
                print(f"Google API error: {e}")
                return []

        links_section = "🔗 Relevant Links from Across the Web:\n\n"
        queries = [
            f"{request.question} Income Tax Act 1961",
            f"{request.question} tax policies India",
            f"{request.question} CBDT guidelines"
        ]

        seen = set()
        total_links = 0
        for query in queries:
            if total_links >= 3:
                break
            results = google_custom_search(query)
            for r in results:
                if r.get('link') not in seen and total_links < 3:
                    seen.add(r.get('link'))
                    total_links += 1
                    summary = llm.invoke(
                        f"Summarize this page for the query '{request.question}':\n"
                        f"Title: {r.get('title')}\nSnippet: {r.get('snippet')}"
                    ).content
                    links_section += f"• {r.get('title')}\n  URL: {r.get('link')}\n  {summary}\n\n"

        return {"answer": links_section.strip()}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

@app.post("/api/data-answer")
async def get_future_data_answer(request: QuestionRequest):
    try:
        if future_error:
            return {"answer": future_error}
        if future_vectorstore is None:
            return {"answer": "Future data not available."}

        llm = get_chatgroq_model(temperature=0.1)

        template = """You are an expert on future tax reforms and proposals as anticipated in 2025. Based only on the following context:
        {context}

        Question: {question}

        Give a futuristic but data-backed explanation. If no context is found, say: "No future data available." """
        prompt_template = ChatPromptTemplate.from_template(template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=future_vectorstore.as_retriever(search_kwargs={'k': 6}),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )

        result = qa_chain({"query": request.question})
        return {"answer": result["result"]}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

@app.post("/api/combined-answer")
async def get_combined_answer(request: QuestionRequest):
    policy_answer = await get_policy_answer(request)
    web_answer = await get_web_answer(request)
    future_answer = await get_future_data_answer(request)
    return {
        "policy_answer": policy_answer["answer"],
        "web_answer": web_answer["answer"],
        "future_answer": future_answer["answer"]
    }

if __name__ == "__main__":
    import uvicorn
    print("\nStarting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)