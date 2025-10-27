#!/usr/bin/env python3
"""
Aider API Wrapper
Provides Python API interface to interact with aider programmatically,
replacing command-line interaction.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from io import StringIO


from aider import models
from aider.coders import Coder
from aider.commands import Commands
from aider.scrape import Scraper, has_playwright
from rich.console import Console

from aider.io import InputOutput
from aider.repo import GitRepo


class APIInputOutput(InputOutput):
    """
    Input/output implementation for programmatic use that mirrors InputOutput behaviour
    while capturing everything in memory instead of displaying it to a terminal.
    """

    def __init__(self, *, auto_confirm: bool = True):
        self.captured_output: List[str] = []
        self.captured_errors: List[str] = []
        self.captured_warnings: List[str] = []
        self.captured_assistant: List[str] = []
        self.user_input_buffer: List[str] = []
        self.chat_transcript: List[str] = []
        self._input_queue: List[str] = []
        self._auto_confirm = auto_confirm

        dummy_input = StringIO()
        dummy_output = StringIO()

        # Initialise the standard InputOutput with non-interactive defaults
        super().__init__(
            pretty=False,
            fancy_input=False,
            input=dummy_input,
            output=dummy_output,
            yes=True if auto_confirm else None,
            notifications=False,
            chat_history_file=None,
            input_history_file=None,
            llm_history_file=None,
        )

        # Replace console with an in-memory buffer so nothing is printed to stdout
        self._console_buffer = StringIO()
        self.console = Console(file=self._console_buffer, force_terminal=False, no_color=True)

    # ------------------------------------------------------------------
    # Helpers
    def _stringify(self, *messages) -> str:
        return " ".join(str(message) for message in messages if message is not None).strip()

    def _append_transcript(self, text: str, linebreak=False, blockquote=False, strip=True):
        if text is None:
            return
        if blockquote:
            if strip:
                text = text.strip()
            text = "> " + text
        if linebreak:
            if strip:
                text = text.rstrip()
            text = text + "  \n"
        if not text.endswith("\n"):
            text += "\n"
        self.chat_transcript.append(text)

    def append_chat_history(self, text, linebreak=False, blockquote=False, strip=True):
        self._append_transcript(text, linebreak=linebreak, blockquote=blockquote, strip=strip)
        super().append_chat_history(text, linebreak=linebreak, blockquote=blockquote, strip=strip)

    # ------------------------------------------------------------------
    # Captured output methods
    def tool_output(self, *messages, log_only=False, bold=False):
        if messages:
            hist = " ".join(str(message) for message in messages).strip()
            if hist:
                self.append_chat_history(hist, linebreak=True, blockquote=True)

        if log_only:
            return

        text = self._stringify(*messages)
        if text:
            self.captured_output.append(text)

    def tool_error(self, message="", strip=True):
        self.num_error_outputs += 1
        text = str(message or "")
        if strip:
            text = text.strip()
        if text:
            self.append_chat_history(text, linebreak=True, blockquote=True)
            self.captured_errors.append(text)

    def tool_warning(self, message="", strip=True):
        text = str(message or "")
        if strip:
            text = text.strip()
        if text:
            self.append_chat_history(text, linebreak=True, blockquote=True)
            self.captured_warnings.append(text)

    def assistant_output(self, message, pretty=None):
        if not message:
            return
        text = str(message)
        self.append_chat_history(text, linebreak=True)
        self.captured_assistant.append(text)

    def user_input(self, inp, log_only=True):
        self.user_input_buffer.append(str(inp))
        hist = str(inp or "").splitlines() or ["<blank>"]
        joined = "  \n#### ".join(hist)
        formatted = "\n#### " + joined
        self.append_chat_history(formatted, linebreak=True)

    # ------------------------------------------------------------------
    # Interaction helpers
    def queue_input(self, value: str):
        """Queue a user input that will be consumed by get_input."""
        self._input_queue.append(value)

    def clear_inputs(self):
        self._input_queue.clear()

    def get_input(
        self,
        root=None,
        rel_fnames=None,
        addable_rel_fnames=None,
        commands=None,
        abs_read_only_fnames=None,
        edit_format=None,
    ):
        return self._input_queue.pop(0) if self._input_queue else ""

    def prompt_ask(self, question, default="", subject=None):
        result = default or ""
        if subject:
            self.tool_output(subject, bold=True)
        self.tool_output(question, log_only=True)
        return result

    def confirm_ask(
        self,
        question,
        default="y",
        subject=None,
        explicit_yes_required=False,
        group=None,
        allow_never=False,
    ):
        self.num_user_asks += 1
        if subject:
            self.tool_output(subject, bold=True)
        self.tool_output(question, log_only=True)
        return True if self._auto_confirm else default.lower().startswith("y")

    def offer_url(self, *args, **kwargs):
        return False

    # ------------------------------------------------------------------
    # Capture management
    def get_captured_output(self):
        """Return the collected interaction logs."""
        return {
            "output": list(self.captured_output),
            "errors": list(self.captured_errors),
            "warnings": list(self.captured_warnings),
            "assistant": list(self.captured_assistant),
            "user_inputs": list(self.user_input_buffer),
            "chat_history": list(self.chat_transcript),
        }

    def clear_captured_output(self):
        """Reset all captured buffers."""
        self.captured_output.clear()
        self.captured_errors.clear()
        self.captured_warnings.clear()
        self.captured_assistant.clear()
        self.user_input_buffer.clear()
        self.chat_transcript.clear()


class AiderAPI:
    """Aider API wrapper for programmatic access"""

    def __init__(self,
                 model: str = "gpt-4",
                 auto_commits: bool = True,
                 git_root: Optional[str] = None,
                 edit_format: Optional[str] = None,
                 stream: bool = False,
                 verify_ssl: bool = True):
        """
        Initialize Aider API instance

        Args:
            model: Model name to use
            auto_commits: Whether to auto-commit changes
            git_root: Git repository root directory
            edit_format: Edit format to use
            stream: Whether to use streaming output
            verify_ssl: Whether to verify SSL certificates for web requests
        """
        self.io = APIInputOutput()
        self.model_name = model
        self.auto_commits = auto_commits
        self.git_root = git_root
        self.edit_format = edit_format
        self.stream = stream
        self.verify_ssl = verify_ssl
        self.coder = None
        self.scraper = None
        self._last_response = None

        self._init_coder()
        self._init_scraper()

    def _init_coder(self):
        """Initialize Coder instance with proper configuration"""
        try:
            # Determine git root directory
            if not self.git_root:
                self.git_root = os.getcwd()

            # Initialize git repository
            repo = None
            if os.path.exists(os.path.join(self.git_root, '.git')):
                repo = GitRepo(self.io, self.git_root, None)

            # Create model instance
            main_model = models.Model(self.model_name)

            # Create Coder instance
            self.coder: Coder = Coder.create(
                main_model=main_model,
                edit_format=self.edit_format,
                io=self.io,
                repo=repo,
                fnames=[],
                read_only_fnames=[],
                show_diffs=False,
                auto_commits=self.auto_commits,
                dirty_commits=self.auto_commits,
                dry_run=False,
                map_tokens=1024,
                verbose=False,
                stream=self.stream,
                use_git=repo is not None,
                restore_chat_history=False,
                auto_lint=False,
                auto_test=False,
                lint_cmds=None,
                test_cmd=None,
                commands=None,
                summarizer=None,
                analytics=None,
                map_refresh=False,
                cache_prompts=False,
                map_mul_no_files=1,
                num_cache_warming_pings=0,
                suggest_shell_commands=False,
                chat_language="en",
                commit_language="en",
                detect_urls=False,
                auto_copy_context=False,
                auto_accept_architect=False,
                add_gitignore_files=False,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize aider: {str(e)}")

    def _init_scraper(self):
        """Initialize web scraper instance"""
        try:
            self.scraper = Scraper(
                print_error=lambda msg: self.io.tool_error(f"Scraper: {msg}"),
                playwright_available=has_playwright(),
                verify_ssl=self.verify_ssl
            )
        except Exception as e:
            self.io.tool_error(f"Failed to initialize web scraper: {str(e)}")
            self.scraper = None

    def scrape_web(self, url: str, use_playwright: Optional[bool] = None) -> Dict[str, Any]:
        """
        Scrape content from a URL and convert to markdown if it's HTML

        Args:
            url: URL to scrape
            use_playwright: Force use playwright (None for auto-detect)

        Returns:
            Scraping result dictionary with content and metadata
        """
        if not self.scraper:
            return {
                'success': False,
                'message': 'Web scraper not initialized',
                'content': None,
                'url': url
            }

        try:
            self.io.clear_captured_output()

            # Temporarily modify playwright setting if specified
            original_playwright = self.scraper.playwright_available
            if use_playwright is not None:
                self.scraper.playwright_available = use_playwright

            content = self.scraper.scrape(url)

            # Restore original setting
            self.scraper.playwright_available = original_playwright

            if content:
                return {
                    'success': True,
                    'message': f"Successfully scraped content from {url}",
                    'content': content,
                    'url': url,
                    'content_length': len(content),
                    'output': self.io.get_captured_output()
                }
            else:
                return {
                    'success': False,
                    'message': f"Failed to retrieve content from {url}",
                    'content': None,
                    'url': url,
                    'output': self.io.get_captured_output()
                }

        except Exception as e:
            return {
                'success': False,
                'message': f"Error scraping {url}: {str(e)}",
                'content': None,
                'url': url,
                'output': self.io.get_captured_output()
            }

    def detect_urls(self, text: str) -> Dict[str, Any]:
        """
        Detect URLs in text content

        Args:
            text: Text to search for URLs

        Returns:
            Dictionary with detected URLs and their positions
        """
        import re

        # URL pattern that matches http, https, ftp and www URLs
        url_pattern = r'(?i)\b((?:https?://|ftp://|www\.)(?:[^\s<>"]+|\([^\s<>"]*\)))'

        try:
            matches = re.finditer(url_pattern, text)
            urls = []

            for match in matches:
                url = match.group(0)
                # Clean up URLs that start with www.
                if url.startswith('www.'):
                    url = 'https://' + url

                urls.append({
                    'url': url,
                    'start': match.start(),
                    'end': match.end(),
                    'matched_text': match.group(0)
                })

            return {
                'success': True,
                'urls': urls,
                'count': len(urls),
                'text_sample': text[:200] + '...' if len(text) > 200 else text
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"Error detecting URLs: {str(e)}",
                'urls': [],
                'count': 0
            }

    def ask_with_web(self, message: str, auto_scrape: bool = True) -> Dict[str, Any]:
        """
        Ask AI a question with automatic web scraping for URLs found in the message

        Args:
            message: User message/question
            auto_scrape: Whether to automatically scrape detected URLs

        Returns:
            AI response result with web content if applicable
        """
        try:
            self.io.clear_captured_output()

            web_content = ""
            scraped_urls = []

            if auto_scrape:
                # Detect URLs in the message
                url_detection = self.detect_urls(message)
                if url_detection['success'] and url_detection['count'] > 0:
                    self.io.tool_output(f"Found {url_detection['count']} URL(s) in message, scraping...")

                    for url_info in url_detection['urls']:
                        url = url_info['url']
                        scrape_result = self.scrape_web(url)

                        if scrape_result['success']:
                            scraped_urls.append(url)
                            # Add scraped content to context
                            web_content += f"\n\n--- Content from {url} ---\n"
                            web_content += scrape_result['content'][:2000]  # Limit to 2000 chars per URL
                            if len(scrape_result['content']) > 2000:
                                web_content += "\n... [content truncated]"
                        else:
                            self.io.tool_error(f"Failed to scrape {url}: {scrape_result['message']}")

            # Combine original message with web content
            enhanced_message = message
            if web_content:
                enhanced_message = message + web_content

            # Execute enhanced message
            self.coder.run_one(enhanced_message, preproc=True)

            # Get response content
            response = self.coder.partial_response_content

            return {
                'success': True,
                'message': "Question processed successfully",
                'response': response,
                'scraped_urls': scraped_urls,
                'url_count': len(scraped_urls),
                'output': self.io.get_captured_output()
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to process question with web content: {str(e)}",
                'response': None,
                'scraped_urls': [],
                'url_count': 0,
                'output': self.io.get_captured_output()
            }

    def web_search_and_ask(self, query: str, search_urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform web search/scrape and then ask AI about the content

        Args:
            query: Question or topic to research
            search_urls: Optional list of URLs to scrape (if None, will look for URLs in query)

        Returns:
            AI response with web research results
        """
        try:
            self.io.clear_captured_output()

            urls_to_scrape = search_urls or []
            scraped_content = []
            scraped_urls = []

            # If no URLs provided, detect them in the query
            if not urls_to_scrape:
                url_detection = self.detect_urls(query)
                if url_detection['success']:
                    urls_to_scrape = [url_info['url'] for url_info in url_detection['urls']]

            # Scrape URLs
            if urls_to_scrape:
                self.io.tool_output(f"Scraping {len(urls_to_scrape)} URL(s) for research...")

                for url in urls_to_scrape:
                    scrape_result = self.scrape_web(url)
                    if scrape_result['success']:
                        scraped_urls.append(url)
                        scraped_content.append(f"Source: {url}\n{scrape_result['content']}")
                    else:
                        self.io.tool_error(f"Failed to scrape {url}: {scrape_result['message']}")

            # Prepare research context
            research_context = ""
            if scraped_content:
                research_context = "\n\n=== WEB RESEARCH ===\n"
                research_context += "\n\n".join(scraped_content)
                research_context += "\n=== END RESEARCH ===\n\n"

            # Enhanced prompt with research context
            enhanced_query = f"""Research Context: {research_context}

User Question: {query}

Please provide a comprehensive answer based on the above research context and your knowledge."""

            # Execute query with research context
            self.coder.run_one(enhanced_query, preproc=True)
            response = self.coder.partial_response_content

            return {
                'success': True,
                'message': "Research and query completed successfully",
                'response': response,
                'scraped_urls': scraped_urls,
                'sources_count': len(scraped_urls),
                'research_context_length': len(research_context),
                'output': self.io.get_captured_output()
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to perform research: {str(e)}",
                'response': None,
                'scraped_urls': [],
                'sources_count': 0,
                'research_context_length': 0,
                'output': self.io.get_captured_output()
            }

    def get_scraper_info(self) -> Dict[str, Any]:
        """
        Get information about the web scraper configuration and capabilities

        Returns:
            Scraper information dictionary
        """
        if not self.scraper:
            return {
                'scraper_initialized': False,
                'playwright_available': False,
                'ssl_verification': self.verify_ssl
            }

        return {
            'scraper_initialized': True,
            'playwright_available': has_playwright(),
            'scraper_playwright_available': self.scraper.playwright_available,
            'ssl_verification': self.verify_ssl,
            'pandoc_available': getattr(self.scraper, 'pandoc_available', None),
            'capabilities': {
                'can_use_playwright': has_playwright(),
                'can_use_httpx': True,
                'can_convert_html_to_markdown': True,
                'can_handle_ssl_errors': not self.verify_ssl
            }
        }

    def add_file(self, file_path: str) -> Dict[str, Any]:
        """
        Add file to context, equivalent to /add command

        Args:
            file_path: File path (relative to git root or absolute path)

        Returns:
            Operation result dictionary
        """
        try:
            self.io.clear_captured_output()
            self.coder.commands.cmd_add(file_path)

            return {
                'success': True,
                'message': f"File '{file_path}' added successfully",
                'output': self.io.get_captured_output(),
                'files_in_context': list(self.coder.abs_fnames)
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to add file '{file_path}': {str(e)}",
                'output': self.io.get_captured_output(),
                'files_in_context': list(self.coder.abs_fnames)
            }

    def remove_file(self, file_path: str) -> Dict[str, Any]:
        """
        Remove file from context, equivalent to /drop command

        Args:
            file_path: File path to remove

        Returns:
            Operation result dictionary
        """
        try:
            self.io.clear_captured_output()
            self.coder.commands.cmd_drop(file_path)

            return {
                'success': True,
                'message': f"File '{file_path}' removed successfully",
                'output': self.io.get_captured_output(),
                'files_in_context': list(self.coder.abs_fnames)
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to remove file '{file_path}': {str(e)}",
                'output': self.io.get_captured_output(),
                'files_in_context': list(self.coder.abs_fnames)
            }

    def ask(self, message: str) -> Dict[str, Any]:
        """
        Ask AI a question and get response

        Args:
            message: User message/question

        Returns:
            AI response result dictionary
        """
        try:
            self.io.clear_captured_output()

            # Execute user message
            self.coder.run_one(message, preproc=True)

            # Get response content
            response = self.coder.partial_response_content

            return {
                'success': True,
                'message': "Question processed successfully",
                'response': response,
                'output': self.io.get_captured_output()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to process question: {str(e)}",
                'response': None,
                'output': self.io.get_captured_output()
            }

    def commit(self, commit_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Git commit, equivalent to /commit command

        Args:
            commit_message: Commit message, if None will auto-generate

        Returns:
            Commit result dictionary
        """
        try:
            self.io.clear_captured_output()
            self.coder.commands.cmd_commit(commit_message)

            return {
                'success': True,
                'message': "Commit successful",
                'output': self.io.get_captured_output()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Commit failed: {str(e)}",
                'output': self.io.get_captured_output()
            }

    def diff(self) -> Dict[str, Any]:
        """
        View current changes, equivalent to /diff command

        Returns:
            Diff result dictionary
        """
        try:
            self.io.clear_captured_output()
            result = self.coder.commands.cmd_diff()

            return {
                'success': True,
                'message': "Diff generated successfully",
                'diff': result,
                'output': self.io.get_captured_output()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to generate diff: {str(e)}",
                'diff': None,
                'output': self.io.get_captured_output()
            }

    def run_command(self, command: str) -> Dict[str, Any]:
        """
        Run any aider command

        Args:
            command: Aider command (e.g., '/add file.py', '/commit', etc.)

        Returns:
            Command execution result dictionary
        """
        try:
            self.io.clear_captured_output()

            # Check if it's a command
            if self.coder.commands.is_command(command):
                result = self.coder.commands.run(command)
            else:
                # If not a command, treat as regular message
                self.coder.run_one(command, preproc=True)
                result = self.coder.partial_response_content

            return {
                'success': True,
                'message': f"Command '{command}' executed successfully",
                'result': result,
                'output': self.io.get_captured_output()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to execute command '{command}': {str(e)}",
                'result': None,
                'output': self.io.get_captured_output()
            }

    def get_files_in_context(self) -> List[str]:
        """
        Get list of files currently in context

        Returns:
            List of file paths
        """
        return [str(f) for f in self.coder.abs_fnames] if self.coder else []

    def get_last_response(self) -> Optional[str]:
        """
        Get the most recent AI response

        Returns:
            Last AI response content
        """
        return self.coder.partial_response_content if self.coder else None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model information

        Returns:
            Model information dictionary
        """
        if not self.coder:
            return {}

        return {
            'model_name': self.coder.main_model.name,
            'edit_format': self.coder.edit_format,
            'max_tokens': getattr(self.coder.main_model, 'max_tokens', None),
            'supports_functions': getattr(self.coder.main_model, 'supports_functions', False)
        }

    def reset_chat(self) -> Dict[str, Any]:
        """
        Reset chat history, equivalent to /reset command

        Returns:
            Reset result dictionary
        """
        try:
            self.io.clear_captured_output()
            self.coder.commands.cmd_reset("")

            return {
                'success': True,
                'message': "Chat history reset successfully",
                'output': self.io.get_captured_output()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to reset chat: {str(e)}",
                'output': self.io.get_captured_output()
            }


# Usage example
if __name__ == "__main__":
    # Create API instance with web scraping enabled
    aider = AiderAPI(model="gpt-4", auto_commits=True, verify_ssl=True)

    # Check scraper capabilities
    print("Checking scraper info...")
    scraper_info = aider.get_scraper_info()
    print(f"Scraper info: {scraper_info}")

    # Basic web scraping
    print("\nScraping web page...")
    scrape_result = aider.scrape_web("https://example.com")
    print(f"Scrape result: {scrape_result['success']}, Content length: {scrape_result.get('content_length', 0)}")

    # URL detection
    print("\nDetecting URLs in text...")
    text_with_urls = "Check out https://github.com and www.openai.com for more info"
    url_detection = aider.detect_urls(text_with_urls)
    print(f"Detected URLs: {url_detection['urls']}")

    # Add file
    print("\nAdding file...")
    result = aider.add_file("example.py")
    print(f"Add result: {result}")

    # Ask question with automatic web scraping
    print("\nAsking question with web scraping...")
    result = aider.ask_with_web("What can you tell me about https://httpbin.org/html?", auto_scrape=True)
    print(f"Ask with web result: {result['success']}, Scraped URLs: {result.get('scraped_urls', [])}")

    # Research specific URLs
    print("\nPerforming web research...")
    research_urls = ["https://httpbin.org/json", "https://httpbin.org/uuid"]
    research_result = aider.web_search_and_ask("What do these endpoints return?", search_urls=research_urls)
    print(f"Research result: {research_result['success']}, Sources: {research_result.get('sources_count', 0)}")

    # Regular ask question
    print("\nAsking regular question...")
    result = aider.ask("Please help me understand this code")
    print(f"Ask result: {result}")

    # Commit changes
    print("\nCommitting changes...")
    result = aider.commit("Update code with AI assistance")
    print(f"Commit result: {result}")
