
import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_embeddings
from src.prompt import system_prompt

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Validate required API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

# Configure Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embedding model
embeddings = download_embeddings()

# Connect to existing Pinecone index
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Retrieve the 3 most relevant document chunks
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Configure Groq chat model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4,
    max_tokens=500,
    groq_api_key=GROQ_API_KEY,
)

# Build the RAG prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the retrieval-augmented generation chain
question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt,
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain,
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a question.", 400

    try:
        response = rag_chain.invoke({"input": msg})
        return str(response["answer"])

    except Exception:
        app.logger.exception("Error while generating chatbot response")
        return "Sorry, something went wrong. Please try again.", 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8080,
        debug=True,
    )
