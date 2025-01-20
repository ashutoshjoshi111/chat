from flask import Blueprint, jsonify, request
import os
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import traceback
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import re
from tqdm import tqdm

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the Flask Blueprint
pdf_reader = Blueprint('pdf_reader', __name__)

# Set OpenAI API Key
openai.api_key = ""

# Global variables to store the vectorstore and retriever
vectorstore = None

# Initialize Pinecone
pinecone = Pinecone(api_key="")


# 6fce9238-ea57-4005-a00b-4a3a0b889253

# Function to create a new Pinecone index if it doesn't exist
def create_pinecone_index():
    if 'pdf-index' not in pinecone.list_indexes().names():
        pinecone.create_index(
            name='pdf-index',
            dimension=1536,  # Embedding dimension for OpenAI embeddings
            metric='cosine',  # Similarity metric
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )


# Normalize the embedding vector
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding  # Avoid division by zero
    return embedding / norm


# Preprocess text by cleaning up extra spaces
def preprocess_text(text):
    # Replace consecutive spaces, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)
    return text


# Process PDF files to extract and chunk the text
def process_pdf(file_path):
    # Create a loader for the PDF
    loader = PyPDFLoader(file_path)
    # Load the document content
    data = loader.load()
    # Split your data into smaller documents with chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    # Extract and return the page content as a list of strings
    texts = [doc.page_content for doc in documents]
    return texts


# Generate an embedding for the input text
def generate_embedding(text):
    try:
        # Make the API call to generate the embedding using the updated OpenAI API
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")

        # Access the embedding from the response object
        embedding = response['data'][0]['embedding']

        # Normalize the embedding
        return normalize_embedding(embedding)

    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None


# Create embeddings for a list of text chunks
def create_embeddings(texts):
    embeddings_list = []
    for text in tqdm(texts, desc="Generating embeddings"):
        try:
            # Generate embedding for the current text
            res = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
            embedding = res['data'][0]['embedding']
            embeddings_list.append(embedding)
        except Exception as e:
            logging.error(f"Error generating embedding for text: {e}")
            embeddings_list.append(None)  # Handle errors gracefully
    return embeddings_list


# Upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings, ids):
    # Ensure embeddings are valid (remove None values)
    valid_embeddings = [(id, embedding) for id, embedding in zip(ids, embeddings) if embedding is not None]
    if valid_embeddings:
        index.upsert(vectors=valid_embeddings)
    else:
        logging.warning("No valid embeddings to upsert.")


# Function to initialize the vectorstore
def initialize_vectorstore():
    global vectorstore
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3800, chunk_overlap=300)
        splits = text_splitter.split_documents(all_docs)

        # Create Pinecone index if it doesn't exist
        create_pinecone_index()

        # Initialize vectorstore
        vectorstore = pinecone.Index("pdf-index")

        # Create and upsert vectors to Pinecone
        for doc in splits:
            embedding = generate_embedding(doc.page_content)
            if embedding is None:
                logging.error(f"Skipping document due to embedding error: {doc.metadata['source']}")
                continue

            vector_id = doc.metadata['source']
            vectorstore.upsert(vectors=[(vector_id, embedding)])


# Query Pinecone for similar vectors
def query_pinecone(query_embedding, top_k=5):
    # Validate the embedding for NaN or infinite values
    if not np.isfinite(query_embedding).all():
        logging.error("Query embedding contains invalid values")
        return None

    # Query Pinecone for the closest vectors
    try:
        query_results = vectorstore.query(queries=[query_embedding], top_k=top_k)
        return query_results
    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}")
        return None


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

        # Generate embedding for the user query
        query_embedding = generate_embedding(user_input)
        if query_embedding is None:
            return jsonify({"error": "Failed to generate query embedding"}), 500

        # Query Pinecone with the generated embedding
        top_k_results = query_pinecone(query_embedding)
        if top_k_results is None:
            return jsonify({"error": "Failed to retrieve results from Pinecone"}), 500

        # Process the query results and return them
        return jsonify({"result": top_k_results})

    except Exception as e:
        logging.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500
