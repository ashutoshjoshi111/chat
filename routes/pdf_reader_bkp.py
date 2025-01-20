from flask import Blueprint, jsonify, request
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import logging
import traceback

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the Flask Blueprint
pdf_reader = Blueprint('pdf_reader', __name__)

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-proj-Q3j6WX1r9BeLTshAlYqLT3BlbkFJLFy6cQXCz5heue01M2wL"

# Global variables to store the vectorstore and retriever
vectorstore = None
retriever = None


# Initialize the vectorstore and retriever once
def initialize_vectorstore():
    global vectorstore, retriever
    if vectorstore is None:
        # Load all PDFs from the directory
        pdf_directory = r"C:\AI\BOTOpenSource\BOTOpenSourceAPI\example_data"
        all_docs = []
        for root, dirs, files in os.walk(pdf_directory):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        for chunk in splits:
            print(chunk)

        # Create vector store and retriever
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()


# Define the GET route
@pdf_reader.route('/', methods=['GET'])
def get_pdf_data():
    try:
        # Initialize the vectorstore only once
        initialize_vectorstore()

        # Validate that the input query exists
        user_input = request.args.get("input", None)
        if not user_input:
            logging.error("No input provided in the request")
            return jsonify({"error": "Input query parameter is missing"}), 400

        # Initialize the language model
        try:
            llm = ChatOpenAI(model="gpt-4o")
        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            return jsonify({"error": "Failed to initialize language model"}), 500

        # Define the system prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        # Create the chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create the question-answering chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Run the chain with the input query
        results = rag_chain.invoke({"input": user_input})

        # Assuming `results` contains non-serializable objects like Documents
        serialized_results = str(results)

        # Return the serialized result as JSON
        return jsonify({"result": serialized_results})

    except Exception as e:
        logging.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500