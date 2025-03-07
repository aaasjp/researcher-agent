import asyncio
from langchain_core.messages import HumanMessage
from react_agent.graph import graph

async def stream_graph_updates(user_input: str, config: dict):
    print(f'User: {user_input}')
    async for event in graph.astream({"messages": [HumanMessage(content=user_input)]},config=config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

    '''
    async for event in graph.astream({"messages": [HumanMessage(content=user_input)]},config=config,stream_mode="values"):
        event["messages"][-1].pretty_print()
    '''
    
        


if __name__ == "__main__":
    user_input = "Hi there! My name is Will."
    config = {"configurable": {"thread_id": "1", "max_search_results": 1}}
    asyncio.run(stream_graph_updates(user_input, config))

    user_input = "Remember my name?"
    asyncio.run(stream_graph_updates(user_input, config))