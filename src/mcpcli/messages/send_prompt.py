from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcpcli.messages.send_message import send_message
from mcpcli.messages.message_types.prompts_messages import PromptsGetMessage

async def send_prompts_get(
    prompt_name: str,
    arguments: dict,
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> list:
    """Send a 'prompts/get' message and return the prompts.
    {
  "method": "prompts/get",
  "params": {
    "name": "analyzeAndRefactorCode",
    "arguments": {}
  }
"""
    message = PromptsGetMessage(prompt_name, arguments)

    # send the message
    response = await send_message(
        read_stream=read_stream,
        write_stream=write_stream,
        message=message,
    )

    # return the result
    return response.get("result", [])
