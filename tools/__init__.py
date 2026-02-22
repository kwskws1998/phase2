"""
Tool system for LLM agent.
Provides tool registration, execution, and management.
"""

from tools.base import Tool, TOOL_REGISTRY, register_tool, get_tool, get_all_tools, get_tools_schema
from tools.executor import execute_tool_call, parse_tool_calls

# Register all tools on import
from tools.biomni import bio_tools
from tools.plan import plan_tools
from tools.vision import vision_tools

__all__ = [
    'Tool',
    'TOOL_REGISTRY',
    'register_tool',
    'get_tool',
    'get_all_tools',
    'get_tools_schema',
    'execute_tool_call',
    'parse_tool_calls',
]
