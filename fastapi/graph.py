# graph.py
from typing import List, TypedDict
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage


############################################################
# 1. Define a Tool
############################################################
@tool
def get_current_weather(city: str) -> str:
    """Simple weather tool."""
    if "london" in city.lower():
        return "The weather in London is 15°C and cloudy."
    elif "tokyo" in city.lower():
        return "The weather in Tokyo is 22°C and sunny."
    return f"No weather info for {city}."


############################################################
# 2. Bind Tools to LLM
############################################################
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools([get_current_weather])


############################################################
# 3. Graph State
############################################################
class AgentState(TypedDict):
    messages: List[AnyMessage]


############################################################
# 4. Nodes
############################################################
def agent_node(state: AgentState):
    """LLM decides: respond or call tool."""
    out = llm_with_tools.invoke(state["messages"])
    return {"messages": [out]}

def tool_node(state: AgentState):
    """Execute tool calls from LLM."""
    last = state["messages"][-1]
    results = []

    for call in last.tool_calls:
        output = get_current_weather.invoke(call["args"])
        results.append(
            ToolMessage(
                content=str(output),
                tool_call_id=call["id"],
            )
        )

    return {"messages": results}


############################################################
# 5. Conditional routing
############################################################
def route(state: AgentState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tool"
    return "end"


############################################################
# 6. Build Graph
############################################################
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges("agent", route, {"tool": "tool", "end": END})
graph.add_edge("tool", "agent")

app = graph.compile()

############################################################
# Export the agent graph
############################################################
agent = app
