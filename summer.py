from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Load & chunk webpage
loader = WebBaseLoader("https://www.chess.com/news/view/announcing-collegiate-chess-league-summer-2025#format")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 2. Generate embeddings & build index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 3. Set up retriever + RAG QA chain
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",  # simplest concatenate approach
    retriever=retriever
)

# 4. Query
query = "What are the main benefits mentioned on the page?"
response = qa.invoke(query)
print(response)
