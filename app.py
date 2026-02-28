from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone index
index_name = "medical-chatbot"
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print(f"Pinecone index '{index_name}' loaded successfully.")
except Exception as e:
    print("Error initializing Pinecone index:", e)
    docsearch = None

if docsearch:
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    chatModel = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
else:
    rag_chain = None


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "No message received", 400

    if not rag_chain:
        return "RAG chain not initialized", 500

    try:
        print("User input:", msg)
        response = rag_chain.invoke({"input": msg})
        print("Full response:", response)
        answer = response.get("answer", "Sorry, I could not find an answer.")
        print("Answer:", answer)
        return str(answer)
    except Exception as e:
        print("Error generating response:", e)
        return f"Internal server error: {e}", 500

if __name__ == "__main__":
    import multiprocessing
    app.run(host="0.0.0.0", port=5050, debug=True)