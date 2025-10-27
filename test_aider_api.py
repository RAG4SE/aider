#!/usr/bin/env python3
"""
Test script for Aider API
Tests the API functionality with file operations and questions
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import aider_api
sys.path.insert(0, str(Path(__file__).parent))

from aider_api import AiderAPI


def test_api():
    """Test the Aider API functionality"""
    print("Testing Aider API...")

    # Change to the test directory
    test_dir = Path("/Users/mac/repo/deepwiki-cli/bench/solidity-interface-demo")
    os.chdir(test_dir)

    # Initialize API with a smaller model for testing
    try:
        aider = AiderAPI(model="deepseek/deepseek-chat", auto_commits=False)
        print("✓ API initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize API: {e}")
        return False

    # Test 1: Check model info
    print("\n1. Getting model info...")
    model_info = aider.get_model_info()
    print(f"Model info: {model_info}")

    # Test 2: Add a file (we'll create DataManager.sol later)
    print("\n2. Testing file operations...")

    # # First, let's see if there are any files in the directory
    # files = list(test_dir.rglob("*"))
    # print(f"Files in test directory: {[f.name for f in files if f.is_file()]}")

    # Test 3: Ask a simple question
    print("\n3. Testing ask functionality...")
    aider.add_file("contracts/DataManager.sol")
    result = aider.ask("Analyze the DataManager.sol file and tell me what is on line 116 (1-based). Only reply the line content.")
    print(f"Ask result success: {result['success']}")
    if result['success']:
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {result['message']}")

    # Test 4: Get files in context
    print("\n4. Getting files in context...")
    files_in_context = aider.get_files_in_context()
    print(f"Files in context: {files_in_context}")

    print("\n✓ API test completed successfully!")
    return True


def test_with_datamanager():
    """Test the API with DataManager.sol file"""
    print("\nTesting with DataManager.sol...")

    # Change to the test directory
    test_dir = Path("/Users/mac/repo/deepwiki-cli/bench/solidity-interface-demo")
    os.chdir(test_dir)

    # Initialize API
    try:
        aider = AiderAPI(model="gpt-3.5-turbo", auto_commits=False)
    except Exception as e:
        print(f"✗ Failed to initialize API: {e}")
        return False

    # Add DataManager.sol file
    print("\n1. Adding DataManager.sol...")
    result = aider.add_file("contracts/DataManager.sol")
    print(f"Add result: {result['success']} - {result['message']}")

    if result['success']:
        print(f"Files in context: {result['files_in_context']}")

        # Ask about line 116
        print("\n2. Asking about line 116...")
        question = "Analyze the DataManager.sol file and tell me what is on line 116 (1-based). Only reply the line content."
        ask_result = aider.ask(question)
        print(f"Ask result success: {ask_result['success']}")
        if ask_result['success']:
            print(f"Response: {ask_result['response']}")
        else:
            print(f"Error: {ask_result['message']}")

    return True


if __name__ == "__main__":
    print("Aider API Test Suite")
    print("=" * 50)

    # Run basic API test
    test_api()

    # Test with DataManager.sol (will be tested after file is created)
    # test_with_datamanager()