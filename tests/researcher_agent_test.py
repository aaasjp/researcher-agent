import asyncio
from react_agent.graph import graph
from langchain_core.messages import HumanMessage

from langgraph_sdk import get_sync_client


client = get_sync_client(url="http://127.0.0.1:2024")
thread = client.threads.create()
config = {"configurable": {"max_search_results": 2}}


for chunk in client.runs.stream(
        thread['thread_id'],  # Threadless run
        "researcher-agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [HumanMessage(content="介绍一下deepseek,并进行总结")],
        },
        config=config,
        stream_mode="values",
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    #print(chunk)
    print("\n\n")





'''
async def main():
    config = {"configurable": {"thread_id": "1", "max_search_results": 1}}
    messages = [HumanMessage(content="介绍一下deepseek,并进行总结")]
    result = await graph.ainvoke({"messages": messages},config)
    print(f'----result:{result}')

if __name__ == "__main__":
    asyncio.run(main())
'''
