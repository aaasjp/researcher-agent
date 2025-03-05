import asyncio
from react_agent.graph import graph
from langchain_core.messages import HumanMessage, SystemMessage

async def main():
    config = {"configurable": {"thread_id": "1", "max_search_results": 1}}
    messages = [HumanMessage(content="介绍一下deepseek")]
    result = await graph.ainvoke({"messages": messages},config)

if __name__ == "__main__":
    asyncio.run(main())