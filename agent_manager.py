"""Agent management and interaction logic."""

import logging
from typing import AsyncIterator
from agents import Agent, Runner, RunItemStreamEvent
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
from tools import (
    search_songs,
    search_songs_by_artist,
    get_collection_stats,
    create_youtube_playlist,
)
from prompts import AGENT_INSTRUCTIONS

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentManager:
    """Manages agent creation and interaction."""

    def __init__(
        self,
        name: str = "VibeCurator Assistant",
        instructions: str = AGENT_INSTRUCTIONS,
        model: str = "gpt-5-mini",
    ):
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=model,
            tools=[
                search_songs,
                search_songs_by_artist,
                get_collection_stats,
                create_youtube_playlist,
            ],
        )

    async def stream_response(
        self, prompt: str, history: list[dict] = None
    ) -> AsyncIterator[dict]:
        """Stream agent response with events including tool calls.

        Args:
            prompt: User input prompt
            history: List of previous messages [{"role": "user", "content": "..."}, ...]

        Yields:
            Dict with 'type' and 'content' keys for text deltas and tool calls
        """
        # Build messages list with history
        messages = []
        if history:
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        result = Runner.run_streamed(self.agent, messages)
        async for event in result.stream_events():
            # Handle text deltas
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                yield {"type": "text", "content": event.data.delta}

            # Handle tool calls
            elif isinstance(event, RunItemStreamEvent) and event.name == "tool_called":
                tool_name = (
                    event.item.raw_item.name
                    if hasattr(event.item, "raw_item")
                    else "Unknown"
                )
                yield {"type": "tool_call", "content": tool_name}

            # Handle tool outputs
            elif isinstance(event, RunItemStreamEvent) and event.name == "tool_output":
                yield {"type": "tool_output", "content": "completed"}

    async def get_response(self, prompt: str, history: list[dict] = None) -> str:
        """Get complete agent response (non-streaming).

        Args:
            prompt: User input prompt
            history: List of previous messages [{"role": "user", "content": "..."}, ...]

        Returns:
            Complete response text
        """
        # Build messages list with history
        messages = []
        if history:
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        result = await Runner.run(self.agent, messages)
        return result.final_output

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.agent.model

    @property
    def agent_name(self) -> str:
        """Get the agent name."""
        return self.agent.name
