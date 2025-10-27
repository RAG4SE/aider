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
                 stream: bool = False):
        """
        Initialize Aider API instance

        Args:
            model: Model name to use
            auto_commits: Whether to auto-commit changes
            git_root: Git repository root directory
            edit_format: Edit format to use
            stream: Whether to use streaming output
        """
        self.io = APIInputOutput()
        self.model_name = model
        self.auto_commits = auto_commits
        self.git_root = git_root
        self.edit_format = edit_format
        self.stream = stream
        self.coder = None
        self._last_response = None

        self._init_coder()

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
            self.coder = Coder.create(
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
            result = self.coder.commands.cmd_add(file_path)

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
            result = self.coder.commands.cmd_drop(file_path)

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
            result = self.coder.commands.cmd_commit(commit_message)

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
    # Create API instance
    aider = AiderAPI(model="gpt-4", auto_commits=True)

    # Add file
    print("Adding file...")
    result = aider.add_file("example.py")
    print(f"Add result: {result}")

    # Ask question
    print("Asking question...")
    result = aider.ask("Please help me understand this code")
    print(f"Ask result: {result}")

    # Commit changes
    print("Committing changes...")
    result = aider.commit("Update code with AI assistance")
    print(f"Commit result: {result}")
