from flask import Flask, request, jsonify, session
from routes.user_routes import user_blueprint  # Import the blueprint from another file
from routes.product_routes import product_blueprint  # Another example blueprint
from routes.pdf_reader import pdf_reader  # Another example blueprint
import os
import bs4
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from flask_session import Session
from datetime import timedelta
import tiktoken  # For counting tokens

#GET http://127.0.0.1:5000/users/
#POST http://127.0.0.1:5000/users/
#GET http://127.0.0.1:5000/products/

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = ""

# Initialize Flask app
app = Flask(__name__)

# Register the blueprints
app.register_blueprint(user_blueprint, url_prefix='/users')
app.register_blueprint(product_blueprint, url_prefix='/products')
app.register_blueprint(pdf_reader, url_prefix='/pdf_reader')

# Configure Flask to use server-side session
app.config["SESSION_TYPE"] = "filesystem"  # This stores session data on the filesystem
app.secret_key = "supersecretkey"  # Necessary for signing session data
Session(app)  # Initialize session handling

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Set session lifetime to 30 minutes
MAX_TOKENS = 4000  # Example for GPT-3.5-turbo

# Initialize the LLM model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Load, chunk, and index the contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Initialize retriever and prompt
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def count_tokens(message, model="gpt-3.5-turbo"):
    """Count the number of tokens in a message using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(message)
    return len(tokens)


def truncate_history(chat_history, max_tokens=MAX_TOKENS):
    total_tokens = 0
    truncated_history = []
    for message in reversed(chat_history):
        message_tokens = count_tokens(message["content"])  # Using custom token counting method
        if total_tokens + message_tokens > max_tokens:
            break
        truncated_history.insert(0, message)  # Insert at the beginning to maintain order
        total_tokens += message_tokens
    return truncated_history


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Hello World"})


@app.route("/hello/<name>", methods=["GET"])
def say_hello(name):
    return jsonify({"message": f"Hello {name}"})


@app.route("/ask", methods=["PUT"])
def ask_question():
    # Get the question from the query parameters
    question = request.args.get("question")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Get or initialize the chat history in session
    if "chat_history" not in session:
        session["chat_history"] = []

    chat_history = session["chat_history"]

    # Add the question to chat history
    chat_history.append({"role": "user", "content": question})

    # Truncate or summarize the chat history to fit the LLM's token limit
    chat_history = truncate_history(chat_history, max_tokens=MAX_TOKENS)  # Ensure token limit

    # Process the question using the rag_chain with truncated chat history
    ai_response = rag_chain.invoke({"input": question, "chat_history": chat_history})

    # Update the chat history with the AI's response
    chat_history.append({"role": "assistant", "content": ai_response["answer"]})

    # Save chat history back to the session
    session["chat_history"] = chat_history

    return jsonify({"response": ai_response["answer"]})


@app.route("/cleanup", methods=["DELETE"])
def cleanup():
    vectorstore.delete_collection()
    session.pop("chat_history", None)  # Clear the session's chat history
    return jsonify({"message": "Collection deleted and chat history cleared"})


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)