# chat_handler.py
import json
import asyncio

from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from mcpcli.llm_client import LLMClient
from mcpcli.system_prompt_generator import SystemPromptGenerator
from mcpcli.tools_handler import convert_to_openai_tools, fetch_tools, handle_tool_call
from mcpcli.messages.send_prompt import send_prompts_get

async def get_input(prompt: str):
    """Get input asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: Prompt.ask(prompt).strip())

async def handle_chat_mode(server_streams, provider="openai", model="gpt-4o-mini"):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    try:
        tools = []
        prompt = ""
        for read_stream, write_stream in server_streams:
            tools.extend(await fetch_tools(read_stream, write_stream))

            response = await send_prompts_get("analyzeAndRefactorCode", {}, read_stream, write_stream)
            prompt = response.get("messages", [])[0].get("content", "").get("text", "") if response else ""
            

        # for (read_stream, write_stream) in server_streams:
        # tools = await fetch_tools(read_stream, write_stream)
        if not tools:
            print("[red]No tools available. Exiting chat mode.[/red]")
            return
        


        system_prompt = generate_system_prompt(tools, prompt)
        openai_tools = convert_to_openai_tools(tools)
        client = LLMClient(provider=provider, model=model)
        conversation_history = [{"role": "system", "content": system_prompt}]

        while True:
            try:
                # Change prompt to yellow
                user_message = await get_input("[bold yellow]>[/bold yellow]")
                if user_message.lower() in ["exit", "quit"]:
                    print(Panel("Exiting chat mode.", style="bold red"))
                    break

                # User panel in bold yellow
                user_panel_text = user_message if user_message else "[No Message]"
                print(Panel(user_panel_text, style="bold yellow", title="You"))

                conversation_history.append({"role": "user", "content": user_message})
                await process_conversation(
                    client, conversation_history, openai_tools, server_streams
                )

            except Exception as e:
                print(f"[red]Error processing message:[/red] {e}")
                continue
    except Exception as e:
        print(f"[red]Error in chat mode:[/red] {e}")


async def process_conversation(
    client, conversation_history, openai_tools, server_streams
):
    """Process the conversation loop, handling tool calls and responses."""
    while True:
        completion = client.create_completion(
            messages=conversation_history,
            tools=openai_tools,
        )

        response_content = completion.get("response", "No response")
        tool_calls = completion.get("tool_calls", [])

        if tool_calls:
            for tool_call in tool_calls:
                # Extract tool_name and raw_arguments as before
                if hasattr(tool_call, "function"):
                    tool_name = getattr(tool_call.function, "name", "unknown tool")
                    raw_arguments = getattr(tool_call.function, "arguments", {})
                elif isinstance(tool_call, dict) and "function" in tool_call:
                    fn_info = tool_call["function"]
                    tool_name = fn_info.get("name", "unknown tool")
                    raw_arguments = fn_info.get("arguments", {})
                else:
                    tool_name = "unknown tool"
                    raw_arguments = {}

                # If raw_arguments is a string, try to parse it as JSON
                if isinstance(raw_arguments, str):
                    try:
                        raw_arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, just display as is
                        pass

                # Now raw_arguments should be a dict or something we can pretty-print as JSON
                tool_args_str = json.dumps(raw_arguments, indent=2)

                tool_md = f"**Tool Call:** {tool_name}\n\n```json\n{tool_args_str}\n```"
                print(
                    Panel(
                        Markdown(tool_md), style="bold magenta", title="Tool Invocation"
                    )
                )

                await handle_tool_call(tool_call, conversation_history, server_streams)
            continue

        # Assistant panel with Markdown
        assistant_panel_text = response_content if response_content else "[No Response]"
        print(
            Panel(Markdown(assistant_panel_text), style="bold blue", title="Assistant")
        )
        conversation_history.append({"role": "assistant", "content": response_content})
        break


def generate_system_prompt(tools, user_system_prompt=""):
    """
    Generate a concise system prompt for the assistant.

    This prompt is internal and not displayed to the user.
    """

    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}

    system_prompt = prompt_generator.generate_prompt(tools_json, user_system_prompt)
    system_prompt += """
**GUIDELINES:**

1. **Reasoning:**
   - Analyze tasks systematically.
   - Break down complex problems.
   - Verify assumptions and reflect on results.

2. **Tool Usage:**
   - **Explore:** Identify and verify information.
   - **Iterate:** Start simple, build on successes, adjust as needed.
   - **Handle Errors:** Analyze, refine approach, document fixes.

3. **Communication:**
   - Explain reasoning and decisions.
   - Share findings transparently.
   - Outline next steps or ask clarifying questions.

**REMINDERS:**
- Ensure each tool call has a clear purpose.
- Make reasonable assumptions when needed.
- Provide actionable insights to minimize user interactions.

**ASSUMPTIONS:**
- Default to descending order if not specified.
"""
    print(system_prompt)
    return system_prompt
