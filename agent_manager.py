"""Agent management and interaction logic."""

from typing import AsyncIterator
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv

load_dotenv()


class AgentManager:
    """Manages agent creation and interaction."""

    def __init__(
        self,
        name: str = "VibeCurator Assistant",
        instructions: str = "You are a helpful assistant specialized in music curation and recommendations.",
        model: str = "gpt-4o",
    ):
        self.agent = Agent(name=name, instructions=instructions, model=model)

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        """Stream agent response token by token.

        Args:
            prompt: User input prompt

        Yields:
            Text deltas as they arrive from the LLM
        """
        result = Runner.run_streamed(self.agent, prompt)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                yield event.data.delta

    async def get_response(self, prompt: str) -> str:
        """Get complete agent response (non-streaming).

        Args:
            prompt: User input prompt

        Returns:
            Complete response text
        """
        result = await Runner.run(self.agent, prompt)
        return result.final_output

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.agent.model

    @property
    def agent_name(self) -> str:
        """Get the agent name."""
        return self.agent.name
