import os
import fitz  # PyMuPDF
import chromadb  # Vector database
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embedding model & ChromaDB
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./vector_db")  # Stores embeddings persistently
collection = chroma_client.get_or_create_collection("pdf_embeddings")

# Initialize session state for conversation history and query
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "query" not in st.session_state:
    st.session_state.query = ""  # Default empty query state
if "first_question_asked" not in st.session_state:
    st.session_state.first_question_asked = False  # Track if first question is asked

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file, splitting into chunks."""
    try:
        doc = fitz.open(pdf_path)
        texts = [page.get_text("text") for page in doc]
        return texts  # List of page-wise text chunks
    except Exception as e:
        return [f"Error extracting text: {e}"]

def store_vectors_in_chromadb(text_chunks):
    """Vectorizes text chunks and stores them in ChromaDB."""
    for i, chunk in enumerate(text_chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(ids=[str(i)], embeddings=[embedding], documents=[chunk])

def retrieve_relevant_chunks(query):
    """Finds all relevant chunks from ChromaDB for the given query."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=len(collection.get()))

    print("DEBUG: Query results:", results)  # Print raw query output

    return results["documents"] if results["documents"] else []


def ask_llm(query, retrieved_text):
    """Uses Gemini AI to generate an answer based on retrieved text."""
    if not retrieved_text or not any(retrieved_text):  # Handle empty case
        return "No relevant information found in the document."
    
    # Debugging output
    print("DEBUG: Retrieved text type:", type(retrieved_text))
    print("DEBUG: Retrieved text content:", retrieved_text)

    # Ensure it's a list of strings
    if isinstance(retrieved_text, list):
        if all(isinstance(item, list) for item in retrieved_text):  # If nested list, flatten it
            retrieved_text = [chunk for sublist in retrieved_text for chunk in sublist]
        elif not all(isinstance(item, str) for item in retrieved_text):  # If elements are not strings, convert them
            retrieved_text = [str(item) for item in retrieved_text]

    context = "\n".join(retrieved_text)  # Now safe to join
    prompt = f"Based on the following document context:\n{context}\nAnswer concisely:\n{query}"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else "No valid response."



def main():
    st.title("PDF RAG-Based Question-Answering App")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        pdf_path = "uploaded_file.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        text_chunks = extract_text_from_pdf(pdf_path)
        store_vectors_in_chromadb(text_chunks)  # Store embeddings in ChromaDB
        st.success("PDF text has been vectorized and stored!")

        # Display conversation history
        st.subheader("Chat History")
        for i, (q, a) in enumerate(st.session_state.conversation_history):
            with st.expander(f"**Q{i+1}:** {q}"):
                st.write(f"**A{i+1}:** {a}")

        # User enters a question
        query = st.text_input("Enter your question:", value=st.session_state.query)  # No key argument here

        if st.button("Ask"):
            if query.strip():  # Check if query is not empty
                relevant_texts = retrieve_relevant_chunks(query)
                answer = ask_llm(query, relevant_texts)
                st.subheader("Answer:")
                st.write(answer)

                # Store the question and answer in the conversation history
                st.session_state.conversation_history.append((query, answer))

                # Set the flag to True once the first question is asked
                if not st.session_state.first_question_asked:
                    st.session_state.first_question_asked = True
            else:
                st.error("Please enter a question.")

        if st.session_state.first_question_asked:
            # Show "Ask New Question" button after the first question is asked
            if st.button("Ask New Question"):
                # Clear the text field for the new question
                st.session_state.query = ""  # Reset the query field to empty
                st.session_state.first_question_asked = False  # Reset flag
                # Optionally, you could choose to clear the conversation history if desired
                # st.session_state.conversation_history = []

if __name__ == "__main__":
    main()