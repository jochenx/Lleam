"""Tool definitions for the Claude API tool runner."""
import json

from anthropic import beta_tool

ALL_TOOLS_JSON = [
    {
        "name": "call_shell",
        "description": "Call shell command, use this to read directories or files. e.g. \"ls -al\" or \"grep -r example_string *.py\" or \"sed -n '21,32p' filename.txt\"",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating it if it doesn't exist or overwriting if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path (supports ~ for home directory)"
                },
                "content": {
                    "type": "string",
                    "description": "The complete content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "str_replace",
        "description": "Replace text in a file by specifying exact old and new strings. This is safer than line-number-based editing as it verifies the exact content before replacing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path (supports ~ for home directory)"
                },
                "old_str": {
                    "type": "string",
                    "description": "The exact string to find and replace (must match exactly including whitespace)"
                },
                "new_str": {
                    "type": "string",
                    "description": "The string to replace it with"
                },
                "occurrence": {
                    "type": "integer",
                    "description": "Which occurrence to replace (1-indexed, -1 for all occurrences)",
                    "default": -1
                }
            },
            "required": ["path", "old_str", "new_str"]
        }
    },
    {
        "name": "final_output",
        "description": "Signal task completion with a detailed summary of work done in markdown format. This tool should ONLY be used in coach mode to signal completion of review and provide feedback.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What was accomplished - a detailed markdown summary of the work done"
                }
            },
            "required": ["summary"]
        }
    }
]

tool_list = json.dumps(ALL_TOOLS_JSON, indent=2, ensure_ascii=True)

# The LLM is sometimes bone-headed and doesn't get it. Use these examples to reinforce a fix.

TOOL_EXAMPLES = {
    "call_shell": [
        '{"name": "call_shell", "input": {"command": "<YOUR COMMAND HERE>"}}',
        '{"name": "call_shell", "input": {"command": "ls -la"}}',
        '{"name": "call_shell", "input": {"command": "grep -r \'TODO\' *.py"}}',
        '{"name": "call_shell", "input": {"command": "cat config.json"}}'
    ],
    "write_file": [
        '{"name": "write_file", "input": {"path": "<FILE PATH>", "content": "<NEW CONTENT>"}}',
        '{"name": "write_file", "input": {"path": "test.py", "content": "print(\'hello\')"}}',
        '{"name": "write_file", "input": {"path": "~/notes.txt", "content": "Meeting at 3pm"}}',
        '{"name": "write_file", "input": {"path": "config.json", "content": "{\\"debug\\": true}"}}'
    ],
    "str_replace": [
        '{"name": "str_replace", "input": {"path": "<FILE PATH>", "old_str": "<CONTENT TO REPLACE>", "new_str": "<NEW CONTENT>"}}',
        '{"name": "str_replace", "input": {"path": "main.py", "old_str": "def old()", "new_str": "def new()"}}',
        '{"name": "str_replace", "input": {"path": "config.py", "old_str": "DEBUG = False", "new_str": "DEBUG = True"}}',
        '{"name": "str_replace", "input": {"path": "test.py", "old_str": "import os", "new_str": "import sys", "occurrence": 1}}'
    ],
    "final_output": [
        '{"name": "final_output", "input": {"summary": "# Task Complete\\n\\nFixed bug in auth module."}}',
        '{"name": "final_output", "input": {"summary": "# Review Complete\\n\\nCode looks good. Approved."}}',
        '{"name": "final_output", "input": {"summary": "# Analysis Done\\n\\nFound 3 issues:\\n- Missing tests\\n- Unused imports\\n- Type hints needed"}}'
    ]
}

collapsed_tool_examples = '\n\n'.join([
    example
    for examples_list in TOOL_EXAMPLES.values()
    for example in examples_list
])

# FWIW, disable_parallel_tool_use=true not part of the steaming API. Even when instructing the LLM to NOT use parallel calls,
# the API calls still fail..... so might as well allow parallel.

TOOL_USE_EXAMPLES = f"""
  **TOOL USE**

  *Tool-First Philosophy*: Solve problems by actively using tools rather than just providing advice.

  If you receive an error when executing a tool call, try again, but fix the call.
  You MUST abandon a tool call if you have attempted the identical call 5 times or more and failed.

  *IMPORTANT*: MAKE SURE TO INCLUDE ALL REQUIRED PARAMETERS. Think very carefully whether you have included them all
  in your tool call, and double check. IF YOU GET AN ERROR BACK, PLEASE FIX YOUR CALL, if you can't figure out 
  how to call a tool, try an alternative, e.g. use the shell command to write contents to a file, similar thing
  goes for str_replace, use a shell command alternative.

  <use_parallel_tool_calls>
  For maximum efficiency, whenever you perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially. Prioritize calling tools in parallel whenever possible. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. When running multiple read-only commands like `ls` or `list_dir`, always run all of the commands in parallel. Err on the side of maximizing parallel tool calls rather than running too many tools sequentially.
  </use_parallel_tool_calls>

  here is the JSON spec for tool use: 
  {tool_list}

  These are examples for some tool calls: 
  {collapsed_tool_examples}
"""


@beta_tool
def call_shell(command: str) -> str:
    """Call shell command, use this to read directories or files.
    e.g. "ls -al" or "grep -r example_string *.py" or "sed -n '21,32p' filename.txt"

    Args:
        command: Shell command to execute
    Returns:
        the output of the shell command
    """
    import subprocess

    try:
        # print first line or abridged...
        first_line = command.split('\n')[0][:80]
        has_more = len(command.split('\n')[0]) > 80 or '\n' in command
        print(f"## running command: {first_line}{'...' if has_more else ''}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )

        # Return stdout if successful, otherwise return stderr
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error (exit code {result.returncode}):\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@beta_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating it if it doesn't exist or overwriting if it does.

    Args:
        path: The file path (supports ~ for home directory)
        content: The complete content to write to the file
    Returns:
        Success or error message
    """
    from pathlib import Path

    try:
        file_path = Path(path).expanduser()

        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content
        file_path.write_text(content)

        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@beta_tool
def str_replace(path: str, old_str: str, new_str: str, occurrence: int = -1) -> str:
    """Replace text in a file by specifying exact old and new strings.

    This is safer than line-number-based editing as it verifies the exact content before replacing.

    Args:
        path: The file path (supports ~ for home directory)
        old_str: The exact string to find and replace (must match exactly including whitespace)
        new_str: The string to replace it with
        occurrence: Which occurrence to replace (1-indexed, -1 for all occurrences, default: -1)
    Returns:
        Success message with details or error message
    """
    from pathlib import Path

    try:
        file_path = Path(path).expanduser()

        if not file_path.exists():
            return f"Error: File {file_path} does not exist"

        # Read current content
        content = file_path.read_text()

        # Check if old_str exists
        if old_str not in content:
            return f"Error: Could not find the specified text in {file_path}"

        # Count occurrences
        count = content.count(old_str)

        # Perform replacement
        if occurrence == -1:
            # Replace all occurrences
            new_content = content.replace(old_str, new_str)
            replaced = count
        elif occurrence > 0 and occurrence <= count:
            # Replace specific occurrence
            parts = content.split(old_str)
            new_content = old_str.join(parts[:occurrence]) + new_str + old_str.join(parts[occurrence:])
            replaced = 1
        else:
            return f"Error: Invalid occurrence {occurrence}. Found {count} occurrence(s)"

        # Write back
        file_path.write_text(new_content)

        return f"Successfully replaced {replaced} occurrence(s) in {file_path}"
    except Exception as e:
        return f"Error replacing text: {str(e)}"

def extract_missing_parameter_info(error_msg: str, content_block) -> str:
    """Extract information about missing parameters from a tool call error.

    Args:
        error_msg: The error message from the failed tool call
        content_block: The tool_use content block that failed

    Returns:
        A helpful message describing which parameters are missing
    """
    tool_params = {
        "call_shell": ["command"],
        "write_file": ["path", "content"],
        "str_replace": ["path", "old_str", "new_str"],
        "run_coverage_report": ["test_command"]
    }

    tool_name = getattr(content_block, 'name', 'unknown')

    if tool_name not in tool_params:
        return "Please provide all required parameters for the tool."

    required_params = tool_params[tool_name]
    provided_params = getattr(content_block, 'input', {})

    missing_params = [param for param in required_params if param not in provided_params]

    if missing_params:
        return f"Tool '{tool_name}' requires these parameters. Missing: {', '.join(missing_params)}"
    else:
        return f"Tool '{tool_name}' requires: {', '.join(required_params)}"


# Global variable to store the last final output
LAST_FINAL_OUTPUT = None
WORK_DIRECTORY = None  # Set by main.py at startup


@beta_tool
def final_output(summary: str) -> str:
    """Signal task completion with a detailed summary of work done in markdown format.

    This tool should ONLY be used in coach mode to signal completion of review and provide feedback.

    Args:
        summary: What was accomplished - a detailed markdown summary of the work done
    Returns:
        Success message
    """
    from pathlib import Path
    import os

    global LAST_FINAL_OUTPUT, WORK_DIRECTORY

    try:
        # Store in global variable
        LAST_FINAL_OUTPUT = summary

        # Use WORK_DIRECTORY if set, otherwise fall back to current directory
        output_dir = WORK_DIRECTORY if WORK_DIRECTORY else Path.cwd()
        output_file = output_dir / "final_output.md"

        # Version number existing final_output.md if it exists
        if output_file.exists():
            # Find the highest existing version number
            version_num = 1
            while True:
                versioned_path = output_dir / f"final_output.md{version_num}"
                if not versioned_path.exists():
                    break
                version_num += 1

            # Rename existing versions in reverse order (highest to lowest)
            for i in range(version_num - 1, 0, -1):
                old_path = output_dir / f"final_output.md{i}"
                new_path = output_dir / f"final_output.md{i + 1}"
                if old_path.exists():
                    old_path.rename(new_path)

            # Rename the current file to version 1
            output_file.rename(output_dir / "final_output.md1")

        # Write the new content
        output_file.write_text(summary)

        return f"Successfully saved final output to {output_file}"
    except Exception as e:
        return f"Error saving final output: {str(e)}"


# List of all available tools (needed for @beta_tool)
# ALL_TOOLS = [call_shell, write_file, str_replace, run_coverage_report, final_output]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name with the given input parameters.

    Args:
        tool_name: The name of the tool to execute
        tool_input: Dictionary of input parameters for the tool

    Returns:
        The result of the tool execution as a string
    """
    tool_map = {
        "call_shell": call_shell,
        "write_file": write_file,
        "str_replace": str_replace,
        "final_output": final_output
    }

    if tool_name not in tool_map:
        return f"Error: Unknown tool '{tool_name}'. Available tools: {', '.join(tool_map.keys())}"

    try:
        tool_func = tool_map[tool_name]
        result = tool_func(**tool_input)
        return result
    except TypeError as e:
        # Handle missing or invalid parameters
        examples = TOOL_EXAMPLES.get(tool_name)
        if examples:
            examples_str = '\n'.join(examples)
            return f"Error calling tool '{tool_name}': {str(e)}\n\n *YOU MUST USE THE COMMAND WITH ALL THE PARAMETERS, FOR EXAMPLE*:\n{examples_str}"
        return f"Error calling tool '{tool_name}': {str(e)}"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"
