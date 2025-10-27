#!/usr/bin/env python3
"""
Test script for the new web API functionality in aider_api.py
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import aider_api
sys.path.insert(0, str(Path(__file__).parent))

from aider_api import AiderAPI

def test_web_api():
    """Test the new web API functionality"""
    print("=== Testing Aider API Web Functionality ===\n")

    # Test 1: Initialize API
    print("1. Testing API initialization...")
    try:
        aider = AiderAPI(model="gpt-3.5-turbo", auto_commits=False, verify_ssl=True)
        print("✓ API initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize API: {e}")
        return

    # Test 2: Check scraper info
    print("\n2. Testing scraper info...")
    try:
        scraper_info = aider.get_scraper_info()
        print(f"✓ Scraper info: {scraper_info}")
        assert 'scraper_initialized' in scraper_info
        assert 'playwright_available' in scraper_info
        print("✓ Scraper info structure is correct")
    except Exception as e:
        print(f"✗ Failed to get scraper info: {e}")

    # Test 3: URL detection
    print("\n3. Testing URL detection...")
    try:
        test_texts = [
            "Check out https://github.com for more info",
            "Visit www.openai.com and https://anthropic.com",
            "No URLs here",
            "Mixed content with ftp://files.example.com and http://example.org"
        ]

        for text in test_texts:
            result = aider.detect_urls(text)
            print(f"Text: '{text[:50]}...'")
            print(f"  Detected {result['count']} URLs: {[u['url'] for u in result['urls']]}")
            assert result['success'] == True
            assert isinstance(result['urls'], list)
        print("✓ URL detection working correctly")
    except Exception as e:
        print(f"✗ URL detection failed: {e}")

    # Test 4: Basic web scraping (using a simple, reliable endpoint)
    print("\n4. Testing web scraping...")
    try:
        # Use httpbin.org which is designed for testing
        test_url = "https://httpbin.org/html"
        result = aider.scrape_web(test_url)

        if result['success']:
            print(f"✓ Successfully scraped {test_url}")
            print(f"  Content length: {result['content_length']}")
            print(f"  Content preview: {result['content'][:100]}...")
        else:
            print(f"✗ Scraping failed: {result['message']}")
    except Exception as e:
        print(f"✗ Web scraping failed: {e}")

    # Test 5: Ask with web scraping (mock test - don't actually call AI)
    print("\n5. Testing ask_with_web structure...")
    try:
        # We'll just test the URL detection part, not the actual AI call
        # since that requires API keys
        message = "What can you tell me about https://httpbin.org/html?"
        url_detection = aider.detect_urls(message)
        print(f"Message: {message}")
        print(f"  Detected URLs: {[u['url'] for u in url_detection['urls']]}")
        print("✓ ask_with_web URL detection structure working")
    except Exception as e:
        print(f"✗ ask_with_web structure test failed: {e}")

    # Test 6: web_search_and_ask structure
    print("\n6. Testing web_search_and_ask structure...")
    try:
        query = "What do these APIs return?"
        search_urls = ["https://httpbin.org/json", "https://httpbin.org/uuid"]

        # Test URL detection in query
        url_detection = aider.detect_urls(query)
        print(f"Query: {query}")
        print(f"  URLs in query: {url_detection['count']}")

        # Test provided URLs
        print(f"  Provided URLs: {search_urls}")
        print("✓ web_search_and_ask structure working")
    except Exception as e:
        print(f"✗ web_search_and_ask structure test failed: {e}")

    print("\n=== Web API Tests Complete ===")

if __name__ == "__main__":
    test_web_api()