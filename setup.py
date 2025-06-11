import os
import subprocess
import sys

def run_command(command):
    process = subprocess.run(command, shell=True, check=True)
    return process.returncode

def main():
    print("🚀 Setting up TDS Virtual TA data collection...")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    run_command("pip install -r requirements.txt")
    
    # Install Playwright browsers
    print("\n🎭 Installing Playwright browsers...")
    run_command("playwright install")
    
    # Create necessary directories
    os.makedirs("downloaded_threads", exist_ok=True)
    os.makedirs("markdown_files", exist_ok=True)
    
    # Run course content scraper
    print("\n📚 Scraping course content...")
    run_command("python scrapers/course_content_scraper.py")
    
    # Run discourse scraper
    print("\n💬 Scraping Discourse posts...")
    run_command("python scrapers/discourse_scraper.py")
    
    # Run preprocessing
    print("\n🔄 Processing scraped data...")
    run_command("python preprocess.py")
    
    print("\n✅ Data collection and processing complete!")

if __name__ == "__main__":
    main()
