#!/usr/bin/env python3
import os
from typing import List, Tuple, Dict, Union
from datetime import datetime

from anthropic import Anthropic
from pathlib import Path

from anthropic.types import Message

from tools import ALL_TOOLS_JSON, get_final_output
from prompts import PROMPTS_SYSTEM_WRITE_PROOF, PROMPTS_SYSTEM_PROMPT_TRANSLATE_PROOF, \
    PROMPTS_SYSTEM_PROMPT_TRANSLATE_VERIFICATION
from utils import load_api_key, read_file

# Create work directory with timestamp
work_dir = Path.home() / "lleam_scratch"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
WORK_DIRECTORY = work_dir / timestamp
WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)

CODE_DIRECTORY = " ~/src/Lleam_generated/test1/MyMathlibProject"
PROOF_FILE = "MyMathlibProject.lean"

# Global configuration
MODEL = "claude-opus-4-5"
MAX_TOKENS = 64_000
THINKING_TOKEN_BUDGET = 20_000

# One strategy for when you run out of tokens is to pass back the partial response and ask for the message to be continued.
# however, for now, assume that any response over 64k tokens is something we don't want to deal with, then just quit.

# Random hard riddle taken from https://www.aivd.nl/onderwerpen/aivd-kerstpuzzel/documenten/publicaties/2025/12/10/aivd-kerstpuzzel-2025
USER_QUESTION = """Please solve this riddle:
Which number fits in place of X?
 3, 1, 8, 5, 18, 188, 2, 28, 23, 7, 27, 88, 238, 38, 278, X"""

REQUIREMENTS = """
    {{REQUIREMENTS}}

    You must translate this question into
    a formal form, then solve it, providing a lean 4 proof for your solution. You have full access to the Mathlib library.
    You MUST attempt to reduce the number of hypotheses to the smallest possible number to make the proof tractable,
    You MUST also attempt to make the proof as expressive as possible, covering as much of the original question as possible. 
    *CRITICAL*: DO NOT INCLUDE THE ORIGINAL QUESTION OR ADD ANY COMMENTS TO YOUR LEAN CODE.
    
    ```""" + USER_QUESTION + """```

    A lean project is set up in """ + CODE_DIRECTORY + """  
    Write your proof in """ + PROOF_FILE


TRANSLATION_INSTRUCTIONS = """ 
    In """ + CODE_DIRECTORY + """ there is a proof in """ + PROOF_FILE + """". Please explain in plain english what it
    proves. Write your output into the """ + CODE_DIRECTORY + """ as 'translation.txt'.
"""

VERIFICATION_INSTRUCTIONS = """
    Read 'translation.txt' in """ + CODE_DIRECTORY + """. Critically evaluate whether the explanation is a valid answer to
    the question ```""" + USER_QUESTION + """```
"""


def update_messages(messages: list, new_message: dict) -> None:
    messages.append(new_message)


def run_llm_query(client: Anthropic, claude_model: str, messages: list, system_prompt: list,
                  message_text: str) -> str:
    """Execute a single task with the Claude API, handling tool calls and errors.

    This function manages the conversation loop with Claude, processing tool calls,
    handling errors with temperature adjustment.

    Args:
        client: Anthropic API client
        claude_model: Model identifier to use
        messages: Conversation history
        system_prompt: System prompt configuration
        message_text: The task message text

    Returns:
        str: Task completion summary or result
    """
    import tools

    # Determine max_tokens
    max_tokens = MAX_TOKENS

    context_window_limit = 200_000
    context_window_remaining = context_window_limit

    total_tokens_used = 0
    tool_call_error = False

    # Temperature control for repeated errors
    temperature = 1 # must be 1 if you use thinking mode
    error_history = []  # List of (tool_name, error_msg) tuples
    REPEATED_ERROR_THRESHOLD = 2
    TOOL_FAILURE_THRESHOLD = 5  # After this many identical failures, disable tools temporarily

    disable_tools_next_call = False

    while True:
        # Check if we need to disable tools due to repeated failures
        if disable_tools_next_call:
            print("## DISABLING TOOLS FOR THIS CALL DUE TO REPEATED FAILURES")
            allow_tools = False
            disable_tools_next_call = False  # reset latch for next loop
        else:
            allow_tools = True

        print(f"#Send messages, len={len(messages)}, temperature={temperature}, tools_enabled={allow_tools}")
        message = send_message(client, messages, system_prompt, claude_model, max_tokens, temperature, allow_tools)

        tool_call_error = False
        last_message = message

        # Check if message is None
        if message is None:
            print("## WARNING: send_message returned None")
            break

        # Calculate token usage
        total_tokens_used = (
                message.usage.input_tokens +
                (message.usage.cache_creation_input_tokens or 0) +
                (message.usage.cache_read_input_tokens or 0) +
                message.usage.output_tokens
        )
        context_window_remaining = context_window_limit - total_tokens_used
        print(f"#Usage: input={message.usage.input_tokens}, output={message.usage.output_tokens}, "
              f"cache_read={message.usage.cache_read_input_tokens or 0}, "
              f"remaining={context_window_remaining:,}")

        # Add assistant message to conversation history
        # Convert content blocks to dicts for JSON serialization
        content_dicts = [
            block.model_dump() if hasattr(block, 'model_dump') else block
            for block in message.content
        ]
        update_messages(messages, {
            "role": "assistant",
            "content": content_dicts
        })

        # Process content blocks and execute tools
        tool_results, has_tool_use, current_errors = process_message_content(
            message.content, temperature, error_history
        )

        # Check if there were any errors in tool execution
        tool_call_error = len(current_errors) > 0

        # Handle error temperature adjustment
        ###jochen - disable for now.
        # temperature, error_history = handle_error_temperature_adjustment(
        #     current_errors, error_history, has_tool_use, temperature, REPEATED_ERROR_THRESHOLD
        # )

        # Check if we've hit the tool failure threshold (5 identical failures)
        if len(error_history) >= TOOL_FAILURE_THRESHOLD:
            recent_errors = error_history[-TOOL_FAILURE_THRESHOLD:]
            if all(e == recent_errors[0] for e in recent_errors):
                print(f"## DETECTED {TOOL_FAILURE_THRESHOLD} IDENTICAL TOOL FAILURES FOR '{recent_errors[0][0]}'")
                disable_tools_next_call = True

                # Get the last error message and examples for this tool
                from tools import TOOL_EXAMPLES
                tool_name = recent_errors[0][0]
                tool_params = recent_errors[0][1]

                examples = TOOL_EXAMPLES.get(tool_name, [])
                examples_str = '\n'.join(examples) if examples else 'No examples available'

                # Add a helpful message about the error
                error_message = f"""You have attempted the same tool call {TOOL_FAILURE_THRESHOLD} times with identical parameters and it has failed each time.
                    Tool: {tool_name}
                    Failed parameters: {tool_params}
                    Please explain what you're trying to do and consider an alternative approach. When I re-enable tools, here are examples of correct usage:
                    {examples_str}
                    What are you trying to accomplish? Let's think about this differently."""

                # Clear error history since we're disabling tools
                error_history = []
                temperature = 1  # Reset temperature too

                # Add the error message
                update_messages(messages, {
                    "role": "user",
                    "content": error_message
                })
                continue

        # If there were tool calls, add the results back to the conversation
        if has_tool_use and tool_results:
            update_messages(messages, {
                "role": "user",
                "content": tool_results
            })
            # Continue the loop to let Claude process the tool results
            print("#Messages after tool results, len=", len(messages))

            continue

        # If there was a tool call error, continue the outer loop to retry
        if tool_call_error:
            continue

        # Check the stop reason after the runner completes
        if last_message is None:
            break

        stop_reason = last_message.stop_reason

        if stop_reason == "max_tokens":
            print("\n## max_tokens stop reason received. Stopping.")
            break

        elif stop_reason == "model_context_window_exceeded":
            print("\n## model_context_window_exceeded stop reason received. Stopping")
            break

        elif stop_reason == "end_turn":
            print("\n## end_turn. (LLM considers job done)")
            return "Task completed"
        else:
            # Normal completion (stop_sequence, etc.)
            break

    # If we exit the loop without proper completion
    return "Task execution incomplete"


def send_message(client: Anthropic, messages: list[dict[str, str | list[dict[str, str | dict[str, str]]]]],
                 system_prompt: list[dict[str, str | dict[str, str]]], claude_model,
                 max_tokens, temperature=1, allow_tools=True) -> Message:
    if allow_tools:
        messages_stream = client.messages.stream(model=claude_model, max_tokens=max_tokens, temperature=temperature,
                                                 system=system_prompt, messages=messages,
                                                 tools=ALL_TOOLS_JSON if allow_tools else None,
                                                 tool_choice={"type": "auto"},
                                                 thinking={
                                                     "type": "enabled",
                                                     "budget_tokens": THINKING_TOKEN_BUDGET
                                                 })
    else:
        messages_stream = client.messages.stream(model=claude_model, max_tokens=max_tokens, temperature=temperature,
                                                 system=system_prompt, messages=messages,
                                                 tool_choice={"type": "none"},
                                                 thinking={
                                                     "type": "enabled",
                                                     "budget_tokens": THINKING_TOKEN_BUDGET
                                                 })

    with messages_stream as stream:
        return stream.get_final_message()

    # I keep getting an internal server error if I try the BetaStreamingToolRunner.


def process_message_content(message_content: list, temperature: float, error_history: list) -> Tuple[
    List[Dict], bool, List[Tuple[str, str]]]:
    """Process message content blocks and execute tools.

    Args:
        message_content: List of content blocks from the message
        temperature: Current temperature setting
        error_history: History of errors for temperature adjustment

    Returns:
        Tuple of (tool_results, has_tool_use, current_errors)
    """
    from tools import execute_tool

    tool_results = []
    has_tool_use = False
    current_errors = []

    for c in message_content:
        if c.type == "thinking":
            print(f"#thinking={c.thinking}")
            # Thinking blocks are informational, no action needed
            
        elif c.type == "text":
            print(f"#text={c.text}")
            # print(f"#text={c.text[:100]}..." if len(c.text) > 100 else f"#text={c.text}")
            # If we got text response (not just tool calls), reset temperature

            ####jochen: for now keep temp at 1
            # if temperature > 0:
            #     print("## Resetting temperature to 0 (got text response)")
            #     # Note: temperature reset is handled by caller

        elif c.type == "tool_use":
            has_tool_use = True
            print(f"#tool_use name={c.name}, input={c.input}")

            # Execute the tool
            try:
                result = execute_tool(c.name, c.input)

                # print first line or abridged...
                first_line = result.split('\n')[0][:80]
                has_more = len(result.split('\n')[0]) > 80 or '\n' in result
                print(f"#tool_result: {first_line}{'...' if has_more else ''}")

                # Add tool result to the list
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": c.id,
                    "content": result
                })

                # Track if result was an error
                if result.startswith("Error"):
                    current_errors.append((c.name, c.input))

            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}"
                print(f"#tool_error: {error_msg}")

                # Track this error
                current_errors.append((c.name, c.input))

                # Add error as tool result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": c.id,
                    "content": error_msg,
                    "is_error": True
                })

    return tool_results, has_tool_use, current_errors


def handle_error_temperature_adjustment(current_errors: List[Tuple[str, str]],
                                        error_history: List[Tuple[str, str]],
                                        has_tool_use: bool,
                                        temperature: float,
                                        repeated_error_threshold: int = 3) -> Tuple[float, List[Tuple[str, str]]]:
    """Handle error tracking and temperature adjustment.

    Args:
        current_errors: Errors from current iteration
        error_history: Historical error list
        has_tool_use: Whether tools were used in this iteration
        temperature: Current temperature
        repeated_error_threshold: Number of repeated errors before adjusting temperature

    Returns:
        Tuple of (new_temperature, updated_error_history)
    """
    new_temperature = temperature
    updated_history = error_history.copy()

    # Check for repeated errors and adjust temperature
    if current_errors:
        updated_history.extend(current_errors)

        # Check if we have repeated similar errors
        if len(updated_history) >= repeated_error_threshold:
            # Check last N errors for similarity
            recent_errors = updated_history[-repeated_error_threshold:]

            # If same tool+params are called multiple times with errors
            if all(e == recent_errors[0] for e in
                   recent_errors):  ### DO NOT CHANGE THIS LINE!!! YOU MUST CHECK BOTH THE TOOL AND PARAMETERS.
                if temperature == 0:
                    print(
                        f"## Detected {repeated_error_threshold} repeated errors for tool '{recent_errors[0][0]}', increasing temperature to 1.0")
                    new_temperature = 1.0
                else:
                    print(f"## Still getting errors for '{recent_errors[0][0]}', keeping temperature at 1.0")

    elif has_tool_use and not current_errors:
        # Had tool use but no errors - reset if needed
        if temperature > 0:
            print("## All tools succeeded, resetting temperature to 0")
            new_temperature = 0.0
            updated_history = []

    # Also reset on text response
    if not has_tool_use and temperature > 0:
        print("## Resetting temperature to 0 (got text response)")
        new_temperature = 0.0
        updated_history = []

    return new_temperature, updated_history



def call_llm(system_prompt_text : str, message_content: str) -> str:
    import tools

    # Set WORK_DIRECTORY in tools module so final_output can use it
    tools.WORK_DIRECTORY = WORK_DIRECTORY

    os.chdir(WORK_DIRECTORY)

    client = Anthropic(api_key=load_api_key(Path.home() / ".llm_keys", "anthropic"))

    # System prompt with cache control
    system_prompt = [
        {
            "type": "text",
            "text": system_prompt_text,
            "cache_control": {"type": "ephemeral"}
        }
    ]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message_content,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]

    print(f"### Model: {MODEL} ###")
    summary = run_llm_query(client, MODEL, messages, system_prompt, REQUIREMENTS)
    print(f"### Task completed: {summary} ###")
    return summary

def main():
    requirements = REQUIREMENTS  # copy the initial value
    done : bool = False
    attemptNo : int = 0
    while attemptNo < 3 and not done:
        attemptNo += 1
        call_llm(PROMPTS_SYSTEM_WRITE_PROOF, requirements)
        call_llm(PROMPTS_SYSTEM_PROMPT_TRANSLATE_PROOF, TRANSLATION_INSTRUCTIONS)
        call_llm(PROMPTS_SYSTEM_PROMPT_TRANSLATE_VERIFICATION, VERIFICATION_INSTRUCTIONS)
        result = get_final_output()
        if "{{{EXPLANATION_ACCEPTED}}}" in result:
            print("### ALL GOOD! ###")
            done = True
        elif "{{{EXPLANATION_REJECTED}}}" in result:
            print("### REJECTED attempt {} ###", attemptNo)
            requirements = REQUIREMENTS + "\n##PREVIOUS ATTEMPTS WERE INSUFFICIENT - PLEASE ADDRESS FEEDBACK\n " + \
                            result + " Attempted proof file: ```" + read_file(CODE_DIRECTORY + '/' + PROOF_FILE) + "```"
        else:
            print("### no answer found :( ###")

    if not done:
        print("### FAILED TO FIND A GOOD SOLUTION, QUITTING ###")

if __name__ == '__main__':
    main()
