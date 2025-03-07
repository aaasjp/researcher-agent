import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-agent-tool"
os.environ["LANGCHAIN_API_KEY"] = "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"

os.environ['OPENAI_API_VERSION']='2024-08-01-preview'
os.environ['AZURE_OPENAI_ENDPOINT']='https://dalle4it.openai.azure.com'
os.environ['AZURE_OPENAI_API_KEY']='627f01bea9194d9b830639bcd5432660'

from langchain_openai import ChatOpenAI
from langchain_openai  import AzureChatOpenAI


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

llm = AzureChatOpenAI  (
    model_name="gpt-4o",
)

# For this ipynb we set parallel tool calling to false as math generally is done sequentially, and this time we have 3 tools that can do math
# the OpenAI model specifically defaults to parallel tool calling for efficiency, see https://python.langchain.com/docs/how_to/tool_calling_parallel/
# play around with it and see how the model behaves with math equations!
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
    print(f'-------------------assistant: state->{state}')
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
print(f'simple-tool-test: messages->{messages}')
messages = react_graph.invoke({"messages": messages})


for m in messages['messages']:
    m.pretty_print()