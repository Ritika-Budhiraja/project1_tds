import os
import json
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup

# === CONFIG ===
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34
CATEGORY_JSON_URL = f"{BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "downloaded_threads")
AUTH_STATE_FILE = os.path.join(OUTPUT_DIR, "auth.json")
POSTS_FILE = os.path.join(OUTPUT_DIR, "discourse_posts.json")
DATE_FROM = datetime(2025, 1, 1)
DATE_TO = datetime(2025, 4, 14)

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

def scrape_posts(playwright):
    print("🔍 Starting scrape using saved session...")
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(storage_state=AUTH_STATE_FILE)
    page = context.new_page()

    try:
        all_topics = []
        page_num = 0
        while True:
            paginated_url = f"{CATEGORY_JSON_URL}?page={page_num}"
            print(f"📦 Fetching page {page_num}...")
            page.goto(paginated_url)

            try:
                data = json.loads(page.inner_text("pre"))
            except:
                data = json.loads(page.content())

            topics = data.get("topic_list", {}).get("topics", [])
            if not topics:
                break

            all_topics.extend(topics)
            page_num += 1

        print(f"📄 Found {len(all_topics)} total topics across all pages")

        filtered_posts = []
        for topic in all_topics:
            created_at = parse_date(topic["created_at"])
            if DATE_FROM <= created_at <= DATE_TO:
                topic_url = f"{BASE_URL}/t/{topic['slug']}/{topic['id']}.json"
                try:
                    page.goto(topic_url)
                    topic_data = json.loads(page.inner_text("pre"))
                except:
                    print(f"⚠️ Failed to fetch topic {topic['id']}, skipping...")
                    continue

                posts = topic_data.get("post_stream", {}).get("posts", [])
                accepted_answer_id = topic_data.get("accepted_answer", topic_data.get("accepted_answer_post_id"))

                reply_counter = {}
                for post in posts:
                    reply_to = post.get("reply_to_post_number")
                    if reply_to is not None:
                        reply_counter[reply_to] = reply_counter.get(reply_to, 0) + 1

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
                    except Exception as e:
                        print(f"⚠️ Failed to process post {post.get('id')}: {e}")
                        continue

        os.makedirs(os.path.dirname(POSTS_FILE), exist_ok=True)
        with open(POSTS_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered_posts, f, indent=2, ensure_ascii=False)

        print(f"✅ Scraped {len(filtered_posts)} posts between {DATE_FROM.date()} and {DATE_TO.date()}")
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
