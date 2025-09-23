import operator
from typing import TypedDict, Annotated, List
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Define the state of our graph
class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    agent_outcome: Annotated[AgentAction, operator.add]
    intermediate_steps: Annotated[List, operator.add]

# Define a tool
@tool
def get_current_weather(location: str):
    """Fetches the current weather for a location."""
    return f"The weather in {location} is 75 degrees and sunny."

# Define the nodes of the graph
def run_llm(state: AgentState):
    """Node that runs the LLM and gets an action."""
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    # The prompt here would be more detailed in a real application
    prompt = f"You are a helpful assistant. Answer the user's question: {state['input']}. Use tools if needed."
    
    # This simulates the LLM's response
    # In a real app, this would be a complex LLM chain
    return {"agent_outcome": AgentFinish(return_values={"output": "The current weather is 75 degrees and sunny."}, log="")}

def execute_tool(state: AgentState):
    """Node that executes the tool."""
    action = state['agent_outcome']
    if action.tool == "get_current_weather":
        return {"intermediate_steps": [(action, get_current_weather.invoke(action.tool_input["location"]))]}
    else:
        return {"intermediate_steps": [(action, "Tool not found.")]}

# Define the conditional edges
def should_continue(state: AgentState):
    """Determines if the agent should continue or finish."""
    if isinstance(state['agent_outcome'], AgentFinish):
        return "end"
    else:
        return "continue"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("llm", run_llm)
workflow.add_node("action", execute_tool)
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)
workflow.add_edge("action", "llm")
workflow.set_entry_point("llm")

# Compile the graph
app = workflow.compile()

# Run the agent
inputs = {"input": "What's the weather in Toronto?"}
result = app.invoke(inputs)

print(result)
