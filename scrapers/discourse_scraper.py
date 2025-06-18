import os
import json
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup
import time

# === CONFIG ===
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34
CATEGORY_JSON_URL = f"{BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "downloaded_threads")
AUTH_STATE_FILE = os.path.join(OUTPUT_DIR, "auth.json")
POSTS_FILE = os.path.join(OUTPUT_DIR, "discourse_posts.json")
TOPICS_PER_PAGE = 30  # Discourse default
POSTS_PER_PAGE = 20   # Discourse default
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

def login_and_save_auth(playwright):
    print("🔐 No auth found. Launching browser for manual login...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    
    try:
        page.goto(f"{BASE_URL}/login")
        print("🌐 Please log in manually using Google. Then press ▶️ (Resume) in Playwright bar.")
        page.pause()
        
        os.makedirs(os.path.dirname(AUTH_STATE_FILE), exist_ok=True)
        context.storage_state(path=AUTH_STATE_FILE)
        print("✅ Login state saved.")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        raise
    finally:
        browser.close()

def is_authenticated(page):
    try:
        page.goto(CATEGORY_JSON_URL, timeout=10000)
        page.wait_for_selector("pre", timeout=5000)
        json.loads(page.inner_text("pre"))
        return True
    except (TimeoutError, json.JSONDecodeError):
        return False
    except Exception as e:
        print(f"❌ Authentication check failed: {e}")
        return False

def fetch_with_retry(page, url, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            page.goto(url)
            try:
                return json.loads(page.inner_text("pre"))
            except:
                return json.loads(page.content())
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Failed to fetch {url} after {max_retries} attempts: {e}")
                raise
            print(f"⚠️ Retry {attempt + 1}/{max_retries} for {url}")
            time.sleep(RETRY_DELAY)

def fetch_all_posts_for_topic(page, topic):
    """Fetch all posts for a topic with pagination"""
    all_posts = []
    post_ids_seen = set()
    
    topic_url = f"{BASE_URL}/t/{topic['slug']}/{topic['id']}.json"
    page_offset = 0
    
    while True:
        paginated_url = f"{topic_url}?page={1}&offset={page_offset}"  # Include offset parameter
        try:
            topic_data = fetch_with_retry(page, paginated_url)
            posts = topic_data.get("post_stream", {}).get("posts", [])
            
            if not posts:
                break
                
            # Add only new posts
            new_posts = [post for post in posts if post["id"] not in post_ids_seen]
            if not new_posts:
                break
                
            for post in new_posts:
                post_ids_seen.add(post["id"])
            all_posts.extend(new_posts)
            
            # Check if we've reached the last page
            if len(all_posts) >= topic_data.get("posts_count", 0):
                break
                
            page_offset += POSTS_PER_PAGE  # Increment offset instead of page number
            time.sleep(0.5)  # Small delay between requests
            
        except Exception as e:
            print(f"⚠️ Error fetching posts at offset {page_offset} for topic {topic['id']}: {e}")
            break
    
    return all_posts, topic_data.get("accepted_answer", topic_data.get("accepted_answer_post_id"))

def scrape_posts(playwright):
    print("🔍 Starting scrape using saved session...")
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(storage_state=AUTH_STATE_FILE)
    page = context.new_page()

    try:
        # Fetch all topics with pagination
        all_topics = []
        page_num = 0
        total_topics_processed = 0
        last_topic_count = 0
        
        while True:
            paginated_url = f"{CATEGORY_JSON_URL}?page={page_num}"
            print(f"📦 Fetching topics page {page_num + 1}...")
            
            data = fetch_with_retry(page, paginated_url)
            topics = data.get("topic_list", {}).get("topics", [])
            
            if not topics:
                break

            total_topics = data.get("topic_list", {}).get("topic_count", 0)  # Get total topics count
            new_topics = [t for t in topics if not any(existing["id"] == t["id"] for existing in all_topics)]
            
            if not new_topics:
                if len(all_topics) >= total_topics:
                    break
            
            all_topics.extend(new_topics)
            total_topics_processed = len(all_topics)
            
            # Check if we're making progress
            if total_topics_processed == last_topic_count:
                if total_topics_processed >= total_topics:
                    break
            
            last_topic_count = total_topics_processed
            print(f"📊 Progress: {total_topics_processed}/{total_topics} topics")
            
            page_num += 1
            time.sleep(0.5)  # Small delay between requests

        print(f"📄 Found {len(all_topics)} total topics")

        # Process each topic and its posts
        filtered_posts = []
        total_posts_processed = 0
        
        for topic_idx, topic in enumerate(all_topics):
            print(f"🔄 Processing topic {topic_idx + 1}/{len(all_topics)}: {topic['title']}")
            
            posts, accepted_answer_id = fetch_all_posts_for_topic(page, topic)
            
            # Build reply counter
            reply_counter = {}
            for post in posts:
                reply_to = post.get("reply_to_post_number")
                if reply_to is not None:
                    reply_counter[reply_to] = reply_counter.get(reply_to, 0) + 1

            # Process posts
            for post in posts:
                try:
                    filtered_posts.append({
                        "topic_id": topic["id"],
                        "topic_title": topic.get("title"),
                        "category_id": topic.get("category_id"),
                        "tags": topic.get("tags", []),
                        "post_id": post["id"],
                        "post_number": post["post_number"],
                        "author": post["username"],
                        "created_at": post["created_at"],
                        "updated_at": post.get("updated_at"),
                        "reply_to_post_number": post.get("reply_to_post_number"),
                        "is_reply": post.get("reply_to_post_number") is not None,
                        "reply_count": reply_counter.get(post["post_number"], 0),
                        "like_count": post.get("like_count", 0),
                        "is_accepted_answer": post["id"] == accepted_answer_id,
                        "mentioned_users": [u["username"] for u in post.get("mentioned_users", [])],
                        "url": f"{BASE_URL}/t/{topic['slug']}/{topic['id']}/{post['post_number']}",
                        "content": BeautifulSoup(post["cooked"], "html.parser").get_text()
                    })
                    total_posts_processed += 1
                    
                    if total_posts_processed % 100 == 0:
                        print(f"💫 Processed {total_posts_processed} posts so far...")
                        
                except Exception as e:
                    print(f"⚠️ Failed to process post {post.get('id')}: {e}")
                    continue

            # Save progress after each topic
            if (topic_idx + 1) % 10 == 0 or topic_idx + 1 == len(all_topics):
                temp_file = os.path.join(OUTPUT_DIR, "discourse_posts_temp.json")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(filtered_posts, f, indent=2, ensure_ascii=False)

        # Save final results
        os.makedirs(os.path.dirname(POSTS_FILE), exist_ok=True)
        with open(POSTS_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered_posts, f, indent=2, ensure_ascii=False)

        print(f"✅ Successfully scraped:")
        print(f"   - {len(all_topics)} topics")
        print(f"   - {len(filtered_posts)} total posts")
        
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        raise
    finally:
        browser.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with sync_playwright() as p:
        if not os.path.exists(AUTH_STATE_FILE):
            login_and_save_auth(p)
        else:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(storage_state=AUTH_STATE_FILE)
            page = context.new_page()
            if not is_authenticated(page):
                print("⚠️ Session invalid. Re-authenticating...")
                browser.close()
                login_and_save_auth(p)
            else:
                print("✅ Using existing authenticated session.")
                browser.close()

        scrape_posts(p)

if __name__ == "__main__":
    main()
