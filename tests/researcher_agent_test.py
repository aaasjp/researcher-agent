import asyncio
from react_agent.graph import graph
from langchain_core.messages import HumanMessage, SystemMessage

async def main():

    messages = [HumanMessage(content="介绍一下deepseek")]
    result = await graph.ainvoke({"messages": messages})

if __name__ == "__main__":
    asyncio.run(main())