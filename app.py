# app.py
import os
import re
import json
import os
import re
import sqlite3
import logging
import numpy as np
import traceback
import aiohttp
import asyncio
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.68  # Lowered threshold for better recall
MAX_RESULTS = 10  # Increased to get more context
load_dotenv()
MAX_CONTEXT_CHUNKS = 4  # Increased number of chunks per source
API_KEY = os.getenv("API_KEY")  # Get API key from environment variable

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key is set
if not API_KEY:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")

# Create a connection to the SQLite database
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Make sure database exists or create it
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create discourse_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS discourse_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER,
        topic_id INTEGER,
        topic_title TEXT,
        post_number INTEGER,
        author TEXT,
        created_at TEXT,
        likes INTEGER,
        chunk_index INTEGER,
        content TEXT,
        url TEXT,
        embedding BLOB
    )
    ''')
    
    # Create markdown_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS markdown_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_title TEXT,
        original_url TEXT,
        downloaded_at TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    conn.close()

# Vector similarity calculation with improved handling
def cosine_similarity(vec1, vec2):
    try:
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return 0 similarity on error rather than crashing

# Function to get embedding from aipipe proxy with retry mechanism
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Getting embedding for text (length: {len(text)})")
            # Call the embedding API through aipipe proxy
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            logger.info("Sending request to embedding API")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["data"][0]["embedding"]
                    else:
                        error_text = await response.text()
                        error_msg = f"Error from embedding API: {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
                        
        except Exception as e:
            error_msg = f"Exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)  # Wait before retry

# Function to find similar content in the database with improved logic
async def find_similar_content(query_embedding, conn):
    try:
        logger.info("Finding similar content in database")
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
        logger.info("Querying discourse chunks")
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
               likes, chunk_index, content, url, embedding 
        FROM discourse_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        discourse_chunks = cursor.fetchall()
        logger.info(f"Processing {len(discourse_chunks)} discourse chunks")
        processed_count = 0
        
        for chunk in discourse_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append({
                        "source": "discourse",
                        "id": chunk["id"],
                        "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"],
                        "topic_title": chunk["topic_title"],
                        "post_number": chunk["post_number"],
                        "author": chunk["author"],
                        "created_at": chunk["created_at"],
                        "likes": chunk["likes"],
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "url": chunk["url"],
                        "similarity": float(similarity)
                    })
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(discourse_chunks)} discourse chunks")
                    
            except Exception as e:
                logger.error(f"Error processing discourse chunk {chunk['id']}: {e}")
        
        # Search markdown chunks
        logger.info("Querying markdown chunks")
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
        FROM markdown_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        markdown_chunks = cursor.fetchall()
        logger.info(f"Processing {len(markdown_chunks)} markdown chunks")
        processed_count = 0
        
        for chunk in markdown_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append({
                        "source": "markdown",
                        "id": chunk["id"],
                        "doc_title": chunk["doc_title"],
                        "original_url": chunk["original_url"],
                        "downloaded_at": chunk["downloaded_at"],
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "similarity": float(similarity)
                    })
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(markdown_chunks)} markdown chunks")

            except Exception as e:
                logger.error(f"Error processing markdown chunk {chunk['id']}: {e}")
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} relevant results above threshold")
        
        # Group by source document and keep most relevant chunks
        grouped_results = {}
        
        for result in results:
            # Create a unique key for the document/post
            if result["source"] == "discourse":
                key = f"discourse_{result['topic_id']}_{result['post_id']}"
            else:
                key = f"markdown_{result['doc_title']}"
            
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].append(result)
        
        # For each source, keep only the most relevant chunks
        final_results = []
        for key, chunks in grouped_results.items():
            # Sort chunks by similarity
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            # Keep top chunks
            final_results.extend(chunks[:MAX_CONTEXT_CHUNKS])
        
        # Sort again by similarity
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top results, limited by MAX_RESULTS
        logger.info(f"Returning {len(final_results[:MAX_RESULTS])} final results after grouping")
        return final_results[:MAX_RESULTS]
    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Function to enrich content with adjacent chunks
async def enrich_with_adjacent_chunks(conn, results):
    try:
        logger.info(f"Enriching {len(results)} results with adjacent chunks")
        cursor = conn.cursor()
        enriched_results = []
        
        for result in results:
            enriched_result = result.copy()
            additional_content = ""
            
            # Try to get adjacent chunks for context
            if result["source"] == "discourse":
                # Get adjacent chunks from the same post
                cursor.execute("""
                    SELECT content 
                    FROM discourse_chunks 
                    WHERE post_id = ? 
                    AND topic_id = ? 
                    AND chunk_index BETWEEN ? AND ?
                    ORDER BY chunk_index
                """, (
                    result["post_id"],
                    result["topic_id"],
                    result["chunk_index"] - 1,
                    result["chunk_index"] + 1
                ))
                adjacent_chunks = cursor.fetchall()
                additional_content = " ".join([chunk["content"] for chunk in adjacent_chunks])
                
            elif result["source"] == "markdown":
                # Get adjacent chunks from the same document
                cursor.execute("""
                    SELECT content 
                    FROM markdown_chunks 
                    WHERE doc_title = ? 
                    AND chunk_index BETWEEN ? AND ?
                    ORDER BY chunk_index
                """, (
                    result["doc_title"],
                    result["chunk_index"] - 1,
                    result["chunk_index"] + 1
                ))
                adjacent_chunks = cursor.fetchall()
                additional_content = " ".join([chunk["content"] for chunk in adjacent_chunks])
            
            # Add the enriched content
            if additional_content:
                enriched_result["content"] = f"{result['content']} {additional_content}"
            
            enriched_results.append(enriched_result)
        
        logger.info(f"Successfully enriched {len(enriched_results)} results")
        return enriched_results
    except Exception as e:
        error_msg = f"Error in enrich_with_adjacent_chunks: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Function to generate an answer using LLM with improved prompt
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:    
        try:
            logger.info(f"Generating answer for question: '{question[:50]}...'")
            context = ""
            for result in relevant_results:
                source_info = ""
                if result["source"] == "discourse":
                    source_info = f"\nSource: {result['topic_title']} (Post #{result['post_number']} by {result['author']})\nURL: {result['url']}\n"
                else:
                    source_info = f"\nSource: {result['doc_title']}\nURL: {result['original_url']}\n"
                
                context += f"{source_info}\nContent: {result['content']}\n\n"
            
            # Prepare improved prompt
            prompt = f"""Answer the following question based ONLY on the provided context. 
            If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Return your response in this exact format:
            1. A comprehensive yet concise answer
            2. A "Sources:" section that lists the URLs and relevant text snippets you used to answer
            
            Sources must be in this exact format:
            Sources:
            1. URL: [exact_url_1], Text: [brief quote or description]
            2. URL: [exact_url_2], Text: [brief quote or description]
            
            Make sure the URLs are copied exactly from the context without any changes.
            """
            
            logger.info("Sending request to LLM API")
            # Call OpenAI API through aipipe proxy
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3  # Lower temperature for more deterministic outputs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(3 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception generating answer: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)  # Wait before retry

# Function to process multimodal content (text + image)
async def process_multimodal_query(question, image_base64):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    try:
        logger.info(f"Processing query: '{question[:50]}...', image provided: {image_base64 is not None}")
        if not image_base64:
            logger.info("No image provided, processing as text-only query")
            return await get_embedding(question)
        
        logger.info("Processing multimodal query with image")
        # Call the GPT-4o Vision API to process the image and question
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Authorization": API_KEY,
            "Content-Type": "application/json"
        }
        
        # Format the image for the API
        image_content = f"data:image/jpeg;base64,{image_base64}"
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Look at this image and tell me what you see related to this question: {question}"},
                        {"type": "image_url", "image_url": {"url": image_content}}
                    ]
                }
            ]
        }
        
        logger.info("Sending request to Vision API")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    # Extract the image analysis from the response
                    image_analysis = result["choices"][0]["message"]["content"]
                    # Combine the image analysis with the original question
                    enhanced_query = f"{question} Context from image: {image_analysis}"
                    # Get embedding for the enhanced query
                    return await get_embedding(enhanced_query)
                else:
                    error_text = await response.text()
                    error_msg = f"Error from Vision API: {error_text}"
                    logger.error(error_msg)
                    # Fall back to text-only query
                    logger.info("Vision API failed, falling back to text-only query")
                    return await get_embedding(question)
    except Exception as e:
        logger.error(f"Exception processing multimodal query: {e}")
        logger.error(traceback.format_exc())
        # Fall back to text-only query
        logger.info("Falling back to text-only query due to exception")
        return await get_embedding(question)

# Function to parse LLM response and extract answer and sources with improved reliability
def parse_llm_response(response):
    try:
        logger.info("Parsing LLM response")
        
        # First try to split by "Sources:" heading
        parts = response.split("Sources:", 1)
        
        # If that doesn't work, try alternative formats
        if len(parts) == 1:
            parts = response.split("References:", 1)
            if len(parts) == 1:
                parts = response.split("Source:", 1)
        
        answer = parts[0].strip()
        links = []
        
        if len(parts) > 1:
            sources_text = parts[1].strip()
            # Look for URLs in the sources section
            url_pattern = r"URL:\s*([^\s,]+)"
            text_pattern = r"Text:\s*([^,\n]+)"
            
            urls = re.finditer(url_pattern, sources_text)
            texts = re.finditer(text_pattern, sources_text)
            
            url_matches = list(urls)
            text_matches = list(texts)
            
            # Pair URLs with their corresponding texts
            for i in range(min(len(url_matches), len(text_matches))):
                url = url_matches[i].group(1).strip("[]")
                text = text_matches[i].group(1).strip("[]")
                links.append(LinkInfo(url=url, text=text))
        
        return answer, links
        
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.error(traceback.format_exc())
        # Return the full response as answer if parsing fails
        return response, []

# Define API routes
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Received query: '{request.question[:50]}...'")
        
        # Get database connection
        conn = get_db_connection()
        
        try:
            # Get embedding for the query
            query_embedding = await process_multimodal_query(request.question, request.image)
            
            # Find similar content
            results = await find_similar_content(query_embedding, conn)
            
            if not results:
                return JSONResponse(
                    status_code=404,
                    content={"answer": "I don't have enough information to answer this question.", "links": []}
                )
            
            # Enrich results with adjacent chunks
            enriched_results = await enrich_with_adjacent_chunks(conn, results)
            
            # Generate answer using LLM
            llm_response = await generate_answer(request.question, enriched_results)
            
            # Parse the response
            answer, links = parse_llm_response(llm_response)
            
            return QueryResponse(answer=answer, links=links)
            
        finally:
            conn.close()
            
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    try:
        # Check database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check tables exist and have data
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy",
            "database": {
                "connected": True,
                "discourse_chunks": discourse_count,
                "markdown_chunks": markdown_count
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)