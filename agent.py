import asyncio
from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv()


async def main():
    agent = Agent(
        name="Assistant", instructions="You are a helpful assistant", model="gpt-5-nano"
    )

    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)

    # Code within the code,
    # Functions calling themselves,
    # Infinite loop's dance.


if __name__ == "__main__":
    asyncio.run(main())
