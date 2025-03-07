from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from react_agent.configuration import Configuration
from react_agent.utils import load_chat_model
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing import Any, Callable, List, Optional, cast, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from langchain_core.language_models import BaseChatModel
from react_agent.utils import load_chat_model
from react_agent.configuration import Configuration
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool



@tool(parse_docstring=True)
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        int: The sum of a and b
    """
    return a + b


@tool(parse_docstring=True)
def mulitply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        int: The product of a and b
    """
    return a * b



@tool(parse_docstring=False)
async def summarize(
    text: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Summarize the given text content.
    
    This function uses a language model to generate a concise summary of the input text.
    It's useful for condensing long articles or documents into key points.
    
    Args:
        text: The text content to be summarized.
        config: The configuration containing the language model.
        
    Returns:
        str: A string containing the summarized content
    """
    configuration = Configuration.from_runnable_config(config)
    llm = load_chat_model(configuration.model)
    
    prompt = f"""请对以下内容进行简明扼要的总结：

{text}

总结："""
    
    response = await llm.ainvoke(prompt)
    return str(response.content)




@tool(parse_docstring=False)
async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


print(convert_to_openai_tool(search)) 