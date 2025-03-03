"""React Agent.

This module defines a custom reasoning and action agent graph.
It invokes tools in a simple loop.
"""

from react_agent.graph import graph
from dotenv import load_dotenv
import os

__all__ = ["graph"]

load_dotenv()
print(f'researcher-agent: __init__.py.os.getenv("AZURE_OPENAI_ENDPOINT")->{os.getenv("AZURE_OPENAI_ENDPOINT")}')
print(f'researcher-agent: __init__.py.os.getenv("OPENAI_API_VERSION")->{os.getenv("OPENAI_API_VERSION")}')
