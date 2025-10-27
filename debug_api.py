#!/usr/bin/env python3
"""
Debug script for Aider API to understand response issue
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import aider_api
sys.path.insert(0, str(Path(__file__).parent))

from aider_api import AiderAPI


def debug_ask():
    """Debug the ask function to see what's happening"""
    print("Debugging Aider API ask method...")

    # Change to the test directory
    test_dir = Path("/Users/mac/repo/deepwiki-cli/bench/solidity-interface-demo")
    os.chdir(test_dir)

    # Initialize API
    try:
        aider = AiderAPI(model="gpt-3.5-turbo", auto_commits=False)
        print("✓ API initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize API: {e}")
        return

    # Check initial state
    print(f"Initial partial_response_content: {repr(aider.coder.partial_response_content)}")

    # Ask a simple question
    print("\nAsking a simple question...")
    result = aider.ask("Hello, can you help me analyze some code?")

    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Response: {repr(result['response'])}")
    print(f"Response type: {type(result['response'])}")

    if result['response']:
        print(f"Response length: {len(result['response'])}")
        print(f"Response content: {result['response']}")

    # Check coder state after ask
    print(f"After ask partial_response_content: {repr(aider.coder.partial_response_content)}")
    print(f"Captured output: {result['output']}")

    # Check if there were any errors
    if result['output']['errors']:
        print(f"Errors captured: {result['output']['errors']}")

    if result['output']['warnings']:
        print(f"Warnings captured: {result['output']['warnings']}")


if __name__ == "__main__":
    debug_ask()