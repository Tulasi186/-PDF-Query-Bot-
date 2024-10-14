from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
#from langchain.chat_models import ChatOpenAI  # Optional, only if needed
from langchain_community.chat_models import ChatOpenAI

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

app = FastAPI()
os.environ["GROQ_API_KEY"] = "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"

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Global variable to store the vectorstore
vectorstore = None

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the uploaded PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text = RecursiveCharacterTextSplitter().split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", 
                                       encode_kwargs={"normalize_embeddings": True})
    vectorstore = FAISS.from_documents(text, embeddings)
    
    return {"message": "PDF uploaded and processed successfully"}

@app.post("/chat")
async def chat(request: ChatRequest):
    global vectorstore
    if vectorstore is None:
        return {"error": "Please upload a PDF first"}
    
    retriever = vectorstore.as_retriever()
    llm = ChatGroq()
    
    template = """
    You are an assistant for question-answering tasks.
    Use the provided context only to answer the following question:
    <context>
    {context}
    </context>
    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    
    response = chain.invoke({"input": request.query})
    return {"response": response["answer"]}

    if __name__ == "__main__":
         import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)


