from typing import Literal
def should_continue_from_llm_node(state) -> Literal["tool_node", "end"]:
    """
    Determine if we should continue from the LLM node
    """
    last_message = state['messages'][-1]    
    if not last_message.tool_calls:
        return "end"
    
    return "tool_node"