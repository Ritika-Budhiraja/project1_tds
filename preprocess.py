# preprocess.py
import os
import sys
import json
import sqlite3
import re
import logging
from bs4 import BeautifulSoup
import html2text
from tqdm import tqdm
import aiohttp
import asyncio
import argparse
import markdown
import time
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Environment variables
API_KEY = os.getenv("API_KEY")
DISCOURSE_API_KEY = os.getenv("DISCOURSE_API_KEY")
DISCOURSE_USERNAME = os.getenv("DISCOURSE_USERNAME")
DISCOURSE_BASE_URL = os.getenv("DISCOURSE_BASE_URL", "https://discourse.onlinedegree.iitm.ac.in")

# Validate environment variables
if not API_KEY:
    logger.error("API_KEY environment variable not set. Please set it in .env file.")
    sys.exit(1)

if not DISCOURSE_API_KEY or not DISCOURSE_USERNAME:
    logger.error("DISCOURSE_API_KEY and DISCOURSE_USERNAME must be set in .env file.")
    sys.exit(1)

logger.info("Environment variables loaded successfully")

# Paths
DISCOURSE_DIR = "downloaded_threads"
MARKDOWN_DIR = "markdown_files"
DB_PATH = "knowledge_base.db"

# Ensure directories exist
os.makedirs(DISCOURSE_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Create a connection to the SQLite database
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        logger.info(f"Connected to SQLite database at {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

# Create the database tables
def create_tables(conn):
    try:
        cursor = conn.cursor()
        
        # Table for Discourse posts chunks
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
        
        # Table for markdown document chunks
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

# Split text into overlapping chunks with improved chunking
def create_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks while preserving sentence boundaries"""
    if not text:
        return []

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Store the current chunk
            chunks.append(' '.join(current_chunk))
            
            # Keep last few sentences for overlap
            overlap_size = 0
            overlap_chunk = []
            
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= chunk_overlap:
                    overlap_chunk.insert(0, s)
                    overlap_size += len(s)
                else:
                    break
                    
            current_chunk = overlap_chunk
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Clean HTML content from Discourse posts
def clean_html(html_content):
    if not html_content:
        return ""
    
    # Use BeautifulSoup to parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    
    # Convert to text and clean up whitespace
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Parse Discourse JSON files
def process_discourse_files(conn):
    cursor = conn.cursor()
    
    # Check if table exists and has data
    cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
    count = cursor.fetchone()[0]
    if count > 0:
        logger.info(f"Found {count} existing discourse chunks in database, skipping processing")
        return
    
    discourse_files = [f for f in os.listdir(DISCOURSE_DIR) if f.endswith('.json')]
    logger.info(f"Found {len(discourse_files)} Discourse JSON files to process")
    
    total_chunks = 0
    
    for file_name in tqdm(discourse_files, desc="Processing Discourse files"):
        try:
            file_path = os.path.join(DISCOURSE_DIR, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Extract topic information
                topic_id = data.get('id')
                topic_title = data.get('title', '')
                topic_slug = data.get('slug', '')
                
                # Process each post in the topic
                posts = data.get('post_stream', {}).get('posts', [])
                
                for post in posts:
                    post_id = post.get('id')
                    post_number = post.get('post_number')
                    author = post.get('username', '')
                    created_at = post.get('created_at', '')
                    likes = post.get('like_count', 0)
                    html_content = post.get('cooked', '')
                    
                    # Clean HTML content
                    clean_content = clean_html(html_content)
                    
                    # Skip if content is too short
                    if len(clean_content) < 20:
                        continue
                    
                    # Create post URL with proper format
                    url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}/{post_number}"
                    
                    # Split content into chunks
                    chunks = create_chunks(clean_content)
                    
                    # Store chunks in database
                    for i, chunk in enumerate(chunks):
                        cursor.execute('''
                        INSERT INTO discourse_chunks 
                        (post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (post_id, topic_id, topic_title, post_number, author, created_at, likes, i, chunk, url, None))
                        total_chunks += 1
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
    
    logger.info(f"Finished processing Discourse files. Created {total_chunks} chunks.")

# Parse markdown files
def process_markdown_files(conn):
    cursor = conn.cursor()
    
    # Check if table exists and has data
    cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
    count = cursor.fetchone()[0]
    if count > 0:
        logger.info(f"Found {count} existing markdown chunks in database, skipping processing")
        return
    
    markdown_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith('.md')]
    logger.info(f"Found {len(markdown_files)} Markdown files to process")
    
    total_chunks = 0
    
    for file_name in tqdm(markdown_files, desc="Processing Markdown files"):
        try:
            file_path = os.path.join(MARKDOWN_DIR, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Extract metadata from frontmatter
                title = ""
                original_url = ""
                downloaded_at = ""
                
                # Extract metadata from frontmatter if present
                frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
                if frontmatter_match:
                    frontmatter = frontmatter_match.group(1)
                    
                    # Extract title
                    title_match = re.search(r'title: "(.*?)"', frontmatter)
                    if title_match:
                        title = title_match.group(1)
                    
                    # Extract original URL
                    url_match = re.search(r'original_url: "(.*?)"', frontmatter)
                    if url_match:
                        original_url = url_match.group(1)
                    
                    # Extract downloaded at timestamp
                    date_match = re.search(r'downloaded_at: "(.*?)"', frontmatter)
                    if date_match:
                        downloaded_at = date_match.group(1)
                    
                    # Remove frontmatter from content
                    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
                
                # Split content into chunks
                chunks = create_chunks(content)
                
                # Store chunks in database
                for i, chunk in enumerate(chunks):
                    cursor.execute('''
                    INSERT INTO markdown_chunks 
                    (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?, NULL)
                    ''', (title, original_url, downloaded_at, i, chunk))
                    total_chunks += 1
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
    
    logger.info(f"Finished processing Markdown files. Created {total_chunks} chunks.")

# Function to create embeddings using aipipe proxy with improved error handling and retries
async def create_embeddings(api_key):
    if not api_key:
        logger.error("API_KEY environment variable not set. Cannot create embeddings.")
        return
        
    conn = create_connection()
    cursor = conn.cursor()
    
    # Get discourse chunks without embeddings
    cursor.execute("SELECT id, content FROM discourse_chunks WHERE embedding IS NULL")
    discourse_chunks = cursor.fetchall()
    logger.info(f"Found {len(discourse_chunks)} discourse chunks to embed")
    
    # Get markdown chunks without embeddings
    cursor.execute("SELECT id, content FROM markdown_chunks WHERE embedding IS NULL")
    markdown_chunks = cursor.fetchall()
    logger.info(f"Found {len(markdown_chunks)} markdown chunks to embed")
    
    # Function to handle long texts by breaking them into multiple embeddings
    async def handle_long_text(session, text, record_id, is_discourse=True, max_retries=3):
        # Model limit is 8191 tokens for text-embedding-3-small
        max_chars = 8000  # Conservative limit to stay under token limit
        
        # If text is within limit, embed it directly
        if len(text) <= max_chars:
            return await embed_text(session, text, record_id, is_discourse, max_retries)
        
        # For long texts, we need to split and create multiple embeddings
        logger.info(f"Text exceeds embedding limit for {record_id}: {len(text)} chars. Creating multiple embeddings.")
        
        # First, get the overlapping subchunks
        overlap = 200  # Small overlap between subchunks for continuity
        subchunks = []
        
        # Create overlapping subchunks
        for i in range(0, len(text), max_chars - overlap):
            end = min(i + max_chars, len(text))
            subchunk = text[i:end]
            if subchunk:
                subchunks.append(subchunk)
        
        logger.info(f"Split into {len(subchunks)} subchunks for embedding")
        
        # Create embeddings for all subchunks
        embeddings = []
        for i, subchunk in enumerate(subchunks):
            logger.info(f"Embedding subchunk {i+1}/{len(subchunks)} for {record_id}")
            success = await embed_text(
                session, 
                subchunk, 
                record_id,
                is_discourse, 
                max_retries,
                f"part_{i+1}_of_{len(subchunks)}"  # Identify this as part of a multi-part embedding
            )
            if not success:
                logger.error(f"Failed to embed subchunk {i+1}/{len(subchunks)} for {record_id}")
        
        return True
    
    # Function to embed a single text with retry mechanism
    async def embed_text(session, text, record_id, is_discourse=True, max_retries=3, part_id=None):
        retries = 0
        while retries < max_retries:
            try:
                # Call the embedding API through aipipe proxy
                url = "https://aipipe.org/openai/v1/embeddings"
                headers = {
                    "Authorization": api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "text-embedding-3-small",
                    "input": text
                }
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result["data"][0]["embedding"]
                        
                        # Convert embedding to binary blob
                        embedding_blob = json.dumps(embedding).encode()
                        
                        # Update the database - handle multi-part embeddings differently
                        if part_id:
                            # For multi-part embeddings, we create additional records
                            if is_discourse:
                                # First, get the original chunk data to duplicate
                                cursor.execute("""
                                SELECT post_id, topic_id, topic_title, post_number, author, created_at, 
                                       likes, chunk_index, content, url FROM discourse_chunks 
                                WHERE id = ?
                                """, (record_id,))
                                original = cursor.fetchone()
                                
                                if original:
                                    # Create a new record with the subchunk and its embedding
                                    cursor.execute("""
                                    INSERT INTO discourse_chunks 
                                    (post_id, topic_id, topic_title, post_number, author, created_at, 
                                     likes, chunk_index, content, url, embedding)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        original["post_id"], 
                                        original["topic_id"], 
                                        original["topic_title"], 
                                        original["post_number"],
                                        original["author"], 
                                        original["created_at"], 
                                        original["likes"], 
                                        f"{original['chunk_index']}_{part_id}",  # Append part_id to chunk_index
                                        text, 
                                        original["url"], 
                                        embedding_blob
                                    ))
                            else:
                                # Handle markdown chunks similarly
                                cursor.execute("""
                                SELECT doc_title, original_url, downloaded_at, chunk_index FROM markdown_chunks 
                                WHERE id = ?
                                """, (record_id,))
                                original = cursor.fetchone()
                                
                                if original:
                                    cursor.execute("""
                                    INSERT INTO markdown_chunks 
                                    (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                    """, (
                                        original["doc_title"],
                                        original["original_url"],
                                        original["downloaded_at"],
                                        f"{original['chunk_index']}_{part_id}",  # Append part_id to chunk_index
                                        text,
                                        embedding_blob
                                    ))
                        else:
                            # For regular embeddings, just update the existing record
                            if is_discourse:
                                cursor.execute(
                                    "UPDATE discourse_chunks SET embedding = ? WHERE id = ?",
                                    (embedding_blob, record_id)
                                )
                            else:
                                cursor.execute(
                                    "UPDATE markdown_chunks SET embedding = ? WHERE id = ?",
                                    (embedding_blob, record_id)
                                )
                        
                        conn.commit()
                        return True
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        logger.error(f"Error embedding text (status {response.status}): {error_text}")
                        return False
            except Exception as e:
                logger.error(f"Exception embedding text: {e}")
                retries += 1
                await asyncio.sleep(3 * retries)  # Wait before retry
        
        logger.error(f"Failed to embed text after {max_retries} retries")
        return False
    
    # Process in smaller batches to avoid rate limits
    batch_size = 10
    async with aiohttp.ClientSession() as session:
        # Process discourse chunks
        for i in range(0, len(discourse_chunks), batch_size):
            batch = discourse_chunks[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, True) for record_id, text in batch]
            results = await asyncio.gather(*tasks)
            logger.info(f"Embedded discourse batch {i//batch_size + 1}/{(len(discourse_chunks) + batch_size - 1)//batch_size}: {sum(results)}/{len(batch)} successful")
            
            # Sleep to avoid rate limits
            if i + batch_size < len(discourse_chunks):
                await asyncio.sleep(2)
        
        # Process markdown chunks
        for i in range(0, len(markdown_chunks), batch_size):
            batch = markdown_chunks[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, False) for record_id, text in batch]
            results = await asyncio.gather(*tasks)
            logger.info(f"Embedded markdown batch {i//batch_size + 1}/{(len(markdown_chunks) + batch_size - 1)//batch_size}: {sum(results)}/{len(batch)} successful")
            
            # Sleep to avoid rate limits
            if i + batch_size < len(markdown_chunks):
                await asyncio.sleep(2)
    
    conn.close()
    logger.info("Finished creating embeddings")

async def fetch_discourse_posts(session, auth_config, start_page=1):
    """Fetch all discourse posts with pagination"""
    base_url = auth_config.get('base_url', 'https://discourse.onlinedegree.iitm.ac.in')
    posts_per_page = 30
    total_posts = 3508
    total_pages = (total_posts + posts_per_page - 1) // posts_per_page
    
    headers = {
        'Api-Key': auth_config['api_key'],
        'Api-Username': auth_config['api_username'],
        'Content-Type': 'application/json'
    }
    
    all_posts = []
    for page in tqdm(range(start_page, total_pages + 1), desc="Fetching posts"):
        try:
            url = f"{base_url}/posts.json?page={page}"
            async with session.get(url, headers=headers) as response:
                if response.status == 429:  # Rate limit
                    wait_time = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                data = await response.json()
                
                if 'latest_posts' in data:
                    posts = data['latest_posts']
                    all_posts.extend(posts)
                    logger.info(f"Fetched {len(posts)} posts from page {page}")
                    
                    # Save progress periodically
                    if page % 10 == 0:
                        save_progress(all_posts, page)
                        
                # Respect rate limits
                await asyncio.sleep(1)
                
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching page {page}: {str(e)}")
            save_progress(all_posts, page - 1)
            raise
            
    return all_posts

def save_progress(posts, last_page):
    """Save fetched posts to avoid losing progress"""
    progress_file = os.path.join(DISCOURSE_DIR, f"posts_progress_page_{last_page}.json")
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved progress to {progress_file}")

# Update the main processing function
async def process_discourse_posts():
    """Process Discourse posts and store in database"""
    specific_posts = [3508]  # Add more post IDs as needed
    
    async with aiohttp.ClientSession() as session:
        posts = await fetch_discourse_posts(session, specific_posts)
        
    conn = create_connection()
    if conn:
        try:
            for post in posts:
                chunks = create_chunks(post['content'], CHUNK_SIZE, CHUNK_OVERLAP)
                for i, chunk in enumerate(chunks):
                    store_discourse_chunk(conn, post, i, chunk)
            conn.commit()
        finally:
            conn.close()
    
def store_discourse_chunk(conn, post, chunk_index, content):
    """Store a chunk of Discourse post in the database"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO discourse_chunks 
            (post_id, topic_id, post_number, author, created_at, likes, chunk_index, content, url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            post['id'],
            post['topic_id'],
            post['post_number'],
            post['author'],
            post['created_at'],
            post['likes'],
            chunk_index,
            content,
            post['url']
        ))
    except sqlite3.Error as e:
        logger.error(f"Error storing discourse chunk: {e}")

# Main function
async def main():
    global CHUNK_SIZE, CHUNK_OVERLAP
    
    parser = argparse.ArgumentParser(description="Preprocess Discourse posts and markdown files for RAG system")
    parser.add_argument("--api-key", help="API key for aipipe proxy (if not provided, will use API_KEY environment variable)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Size of text chunks (default: {CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"Overlap between chunks (default: {CHUNK_OVERLAP})")
    args = parser.parse_args()
    
    # Get API key from arguments or environment variable
    api_key = args.api_key or API_KEY
    if not api_key:
        logger.error("API key not provided. Please provide it via --api-key argument or API_KEY environment variable.")
        return
    
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    
    logger.info(f"Using chunk size: {CHUNK_SIZE}, chunk overlap: {CHUNK_OVERLAP}")
    
    # Create database connection
    conn = create_connection()
    if conn is None:
        return
    
    # Create tables
    create_tables(conn)
    
    # Process files
    process_discourse_files(conn)
    process_markdown_files(conn)
    
    # Create embeddings
    await create_embeddings(api_key)
    
    # Close connection
    conn.close()
    logger.info("Preprocessing complete")

if __name__ == "__main__":
    try:
        # Create database tables
        conn = create_connection()
        if conn:
            create_tables(conn)
            conn.close()
        
        # Run async processing
        asyncio.run(process_discourse_posts())
        logger.info("Successfully processed Discourse posts")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")