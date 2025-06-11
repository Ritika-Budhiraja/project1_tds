# preprocess.py
import os
import re
import json
import logging
import sqlite3
import argparse
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
DISCOURSE_DIR = "downloaded_threads"
MARKDOWN_DIR = "markdown_files"
CHUNK_SIZE = 1000  # Size of text chunks
CHUNK_OVERLAP = 200  # Overlap between chunks

# Configuration that can be modified via command line args
config = {
    'chunk_size': CHUNK_SIZE,  # Size of text chunks
    'chunk_overlap': CHUNK_OVERLAP  # Overlap between chunks
}

# Load environment variables
load_dotenv(override=True)  # Force reload of environment variables
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY environment variable not set. Please set it before running.")
else:
    logger.info("API key loaded successfully")

def clean_html(html_content):
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_chunks(text, chunk_size, chunk_overlap):
    if not text:
        return []
    
    chunks = []
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [text]
    
    current_pos = 0
    while current_pos < len(text):
        chunk_end = min(current_pos + chunk_size, len(text))
        if chunk_end < len(text):
            # Try to find a sentence boundary
            last_period = text.rfind('.', current_pos, chunk_end)
            if last_period != -1 and last_period > current_pos:
                chunk_end = last_period + 1
        
        chunk = text[current_pos:chunk_end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move to next chunk position, considering overlap
        current_pos = chunk_end - chunk_overlap if chunk_end < len(text) else chunk_end
    
    return chunks

def create_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        logger.info(f"Connected to SQLite database at {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def create_tables(conn):
    try:
        cursor = conn.cursor()
        cursor.execute('''
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
        
        cursor.execute('''
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
        logger.info("Database tables created successfully")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")

def process_discourse_files(conn, chunk_size=1000, chunk_overlap=200):
    cursor = conn.cursor()
    total_chunks = 0
    
    try:
        # First check if we already have processed chunks
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        count = cursor.fetchone()[0]
        if count > 0:
            logger.info(f"Found {count} existing discourse chunks in database, skipping processing")
            return
          # Find all JSON files except auth.json
        discourse_files = [f for f in os.listdir(DISCOURSE_DIR) if f.endswith('.json') and f != 'auth.json']
        logger.info(f"Found {len(discourse_files)} Discourse JSON files to process")
        
        for file_name in tqdm(discourse_files, desc="Processing Discourse files"):
            try:
                file_path = os.path.join(DISCOURSE_DIR, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    posts = json.load(file)
                    logger.info(f"Found {len(posts)} posts in {file_name}")
                    for post in posts:
                        # Extract fields using the correct JSON structure
                        topic_id = post.get('topic_id')
                        topic_title = post.get('topic_title', '')
                        post_id = post.get('post_id')
                        post_number = post.get('post_number', 1)
                        author = post.get('author', '')
                        created_at = post.get('created_at', '')
                        likes = post.get('like_count', 0)
                        url = post.get('url', '')
                        content = post.get('content', '')
                        
                        # Skip if content is too short
                        if not content or len(content) < 20:
                            continue
                        
                        # Create chunks from the content
                        chunks = create_chunks(content, chunk_size, chunk_overlap)
                        
                        # Insert each chunk into the database
                        for i, chunk in enumerate(chunks):
                            cursor.execute('''
                            INSERT INTO discourse_chunks 
                            (post_id, topic_id, topic_title, post_number, author, 
                             created_at, likes, chunk_index, content, url, embedding)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (post_id, topic_id, topic_title, post_number, author, 
                                  created_at, likes, i, chunk, url, None))
                            total_chunks += 1
                            
                            # Commit every 100 chunks to avoid transaction overhead
                            if total_chunks % 100 == 0:
                                conn.commit()
                
                # Commit remaining chunks
                conn.commit()
                logger.info(f"Processed {total_chunks} chunks from file {file_name}")
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}")
                continue
        
        logger.info(f"Finished processing Discourse files. Created {total_chunks} chunks.")
    except Exception as e:
        logger.error(f"Error processing Discourse files: {str(e)}")
        raise

async def create_embeddings(api_key):
    if not api_key:
        logger.error("API_KEY environment variable not set. Cannot create embeddings.")
        return
        
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, content FROM discourse_chunks WHERE embedding IS NULL")
    discourse_chunks = cursor.fetchall()
    logger.info(f"Found {len(discourse_chunks)} discourse chunks to embed")
    
    async with aiohttp.ClientSession() as session:
        batch_size = 10
        for i in range(0, len(discourse_chunks), batch_size):
            batch = discourse_chunks[i:i+batch_size]
            for record_id, text in batch:
                url = "https://aipipe.org/openai/v1/embeddings"
                headers = {
                    "Authorization": api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "text-embedding-3-small",
                    "input": text
                }
                
                try:
                    async with session.post(url, headers=headers, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            embedding = result["data"][0]["embedding"]
                            embedding_blob = json.dumps(embedding).encode()
                            
                            cursor.execute(
                                "UPDATE discourse_chunks SET embedding = ? WHERE id = ?",
                                (embedding_blob, record_id)
                            )
                            conn.commit()
                            logger.info(f"Successfully embedded chunk {record_id}")
                        else:
                            error_text = await response.text()
                            logger.error(f"Error embedding chunk {record_id}: {error_text}")
                except Exception as e:
                    logger.error(f"Exception embedding chunk {record_id}: {e}")
            
            if i + batch_size < len(discourse_chunks):
                await asyncio.sleep(2)
    
    conn.close()
    logger.info("Finished creating embeddings")

async def main():
    parser = argparse.ArgumentParser(description="Preprocess Discourse posts and markdown files for RAG system")
    parser.add_argument("--api-key", help="API key for aipipe proxy (if not provided, will use API_KEY environment variable)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks (default: 200)")
    args = parser.parse_args()
    
    api_key = args.api_key or API_KEY
    if not api_key:
        logger.error("API key not provided. Please provide it via --api-key argument or API_KEY environment variable.")
        return
    
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    logger.info(f"Using chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")
    
    conn = create_connection()
    if conn is None:
        return
    try:
        create_tables(conn)
        process_discourse_files(conn, chunk_size, chunk_overlap)
        await create_embeddings(api_key)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        conn.close()
        logger.info("Preprocessing complete")

if __name__ == "__main__":
    asyncio.run(main())
