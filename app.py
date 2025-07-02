import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from rank_bm25 import BM25Okapi
from groq import Groq
import numpy as np
import os
import warnings
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tempfile
import uvicorn
from dotenv import load_dotenv

load_dotenv(override=True)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

class ChatMessage(BaseModel):
    message: str

class GlobalRAGSystem:
    def __init__(self):
        self.rag_chat = None
        self.is_ready = False
        self.model = None

global_rag = GlobalRAGSystem()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        full_text += page.get_text()
    doc.close()
    return full_text

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def generate_embeddings(chunks, model_name="BAAI/bge-small-en-v1.5"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embeddings, model

def setup_vector_store(chunks, embeddings, collection_name="rag_documents"):
    client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name)
    ids = [f"chunk_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": i, "source": "document.pdf"} for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=chunks)
    return client, collection

def setup_advanced_retrieval(collection, chunks, model):
    # Precompute BM25 for keyword matching
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    def advanced_search(query_text, n_results=10):
        # Generate query embedding
        query_embedding = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Semantic search with higher initial results
        dense_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results * 2, len(chunks)),  # Get more candidates
            include=["documents", "metadatas", "distances"]
        )
        
        # BM25 keyword search
        query_tokens = query_text.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Advanced hybrid scoring with multiple factors
        results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            dense_results['documents'][0],
            dense_results['metadatas'][0], 
            dense_results['distances'][0]
        )):
            chunk_id = metadata['chunk_id']
            
            # Semantic similarity (0-1, higher is better)
            semantic_score = 1 - distance
            
            # BM25 score (normalized)
            bm25_score = bm25_scores[chunk_id] if chunk_id < len(bm25_scores) else 0
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            normalized_bm25 = bm25_score / max_bm25
            
            # Query term overlap bonus
            doc_tokens = set(doc.lower().split())
            query_token_set = set(query_tokens)
            overlap_ratio = len(doc_tokens.intersection(query_token_set)) / len(query_token_set) if query_token_set else 0
            
            # Length penalty (prefer chunks that aren't too short)
            length_score = min(len(doc) / 500, 1.0)  # Normalize around 500 chars
            
            # Final hybrid score with weighted combination
            final_score = (
                0.5 * semantic_score +      # Semantic similarity (50%)
                0.25 * normalized_bm25 +    # Keyword matching (25%)
                0.15 * overlap_ratio +      # Direct term overlap (15%)
                0.1 * length_score          # Content length (10%)
            )
            
            results.append({
                'document': doc,
                'score': final_score,
                'metadata': metadata,
                'chunk_id': chunk_id,
                'semantic_score': semantic_score,
                'bm25_score': normalized_bm25
            })
        
        # Sort by final score and apply stricter filtering
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Dynamic threshold - only include chunks significantly relevant
        if results:
            top_score = results[0]['score']
            threshold = max(0.3, top_score * 0.6)  # Adaptive threshold
            filtered_results = [r for r in results if r['score'] >= threshold]
            return filtered_results[:n_results]
        
        return results[:n_results]
    
    return advanced_search

def setup_rag_generation(groq_api_key, search_fn):
    client = Groq(api_key=groq_api_key)
    
    def rag_chat(user_query, n_context_chunks=6, temperature=0.05, max_tokens=400):
        # Get relevant contexts
        search_results = search_fn(user_query, n_results=n_context_chunks)
        
        if not search_results:
            return {
                'query': user_query,
                'answer': "I couldn't find relevant information in the document to answer your question. Please try rephrasing or ask about different topics.",
                'tokens_used': 0,
                'cost': 0
            }
        
        # Build context with only the most relevant chunks
        contexts = []
        for result in search_results:
            if result['score'] > 0.2:  # Only high-quality matches
                contexts.append(result['document'])
        
        if not contexts:
            return {
                'query': user_query,
                'answer': "I couldn't find sufficiently relevant information in the document. Please try a more specific question.",
                'tokens_used': 0,
                'cost': 0
            }
        
        combined_context = "\n\n".join(contexts)
        
        # Optimized system prompt for direct, accurate responses
        system_prompt = """You are a precise AI assistant that answers questions based strictly on the provided document context.

CRITICAL RULES:
1. Answer directly and naturally - NO mentions of "context", "document", or "provided information"
2. If the context contains the answer, provide it clearly and completely
3. If the context is insufficient, say "I don't have enough information about that topic"
4. Be conversational but accurate - write as if you naturally know this information
5. Never mention relevance scores, sources, or that you're using context
6. Start immediately with the answer - no preamble"""
        
        user_prompt = f"""Context information:
{combined_context}

Question: {user_query}

Provide a direct, natural answer based on the information above:"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.8
        )
        
        return {
            'query': user_query,
            'answer': response.choices[0].message.content.strip(),
            'tokens_used': response.usage.total_tokens,
            'cost': (response.usage.total_tokens * 0.27) / 1_000_000
        }
    
    return rag_chat

def build_rag_system(pdf_path, groq_api_key):
    print(f"Building RAG system for: {pdf_path}")
    
    # Extract text
    extracted_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(extracted_text)} characters")
    
    # Create optimized chunks
    chunks = chunk_text(extracted_text)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    embeddings, model = generate_embeddings(chunks)
    print(f"Generated embeddings: {embeddings.shape}")
    
    # Store model globally
    global_rag.model = model
    
    # Setup vector store
    client, collection = setup_vector_store(chunks, embeddings)
    print(f"Vector store created with {collection.count()} documents")
    
    # Setup advanced retrieval
    search_fn = setup_advanced_retrieval(collection, chunks, model)
    
    # Setup RAG generation
    rag_chat = setup_rag_generation(groq_api_key, search_fn)
    
    print("RAG system build complete!")
    return rag_chat

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found</h1>"

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    print(f"Received file upload: {file.filename}")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        print(f"Temporary file saved: {tmp_file_path}")
        
        # Build RAG system
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
        
        rag_chat = build_rag_system(tmp_file_path, api_key)
        
        # Update global system
        global_rag.rag_chat = rag_chat
        global_rag.is_ready = True
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        print("PDF processing completed successfully")
        return {
            "status": "success",
            "message": "PDF processed successfully! Ready for chat."
        }
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat")
async def chat(message: ChatMessage):
    print(f"Received chat message: {message.message}")
    
    if not global_rag.is_ready or not global_rag.rag_chat:
        raise HTTPException(status_code=400, detail="Please upload a PDF first")
    
    try:
        result = global_rag.rag_chat(message.message)
        print(f"Chat response generated, cost: ${result['cost']:.6f}")
        return {
            "answer": result['answer'],
            "cost": result['cost']
        }
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

if __name__ == "__main__":
    print("Starting Gravitas AI RAG System...")
    print("Make sure to set GROQ_API_KEY environment variable")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")