import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract text from all paragraph tags
        paragraphs = soup.find_all("p")
        text = "\n".join([para.get_text() for para in paragraphs if para.get_text()])
        return text if text else "No readable content found on the page."
    except Exception as e:
        return f"Error fetching URL: {str(e)}"
    
url = "https://en.wikipedia.org/wiki/Keanu_Reeves"
text = extract_text_from_url(url)

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Convert the extracted text into documents
source_docs = [Document(page_content=text, metadata={"source": url})]

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(docs_processed, embeddings)
retriever = db.as_retriever()

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192", verbose=True)

from langchain import PromptTemplate
prompt_template = PromptTemplate(
    template="You are a helpful assistant. Based on the following context:\n\n{context}\n\nAnswer this question:\n{question}",
    input_variables=["context", "question"]
)

retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
)

retrievalQA.invoke({"query":"Who is mother of Keanu Reeves"})