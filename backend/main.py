'''
Need to set up the langchain
Need to pull data from bucket
Need to upload data to bucket


Front-end: need to give the data to display
'''
from langchain_fireworks import Fireworks 
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages 

from typing import TypedDict, Annotated
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os 

load_dotenv()

# Define the state
class SummaryState(TypedDict):
    research_paper: str
    draft_summary: str
    refined_summary: str
    evaluation: str

def generate_response(prompt):
    llm = Fireworks(
        api_key= os.getenv("FIREWORKS_API_KEY"),
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        max_tokens=256)
    llm(prompt)


# Define the nodes
def draft_summary(state: SummaryState) -> SummaryState:
    # Simulate drafting a summary
    full_paper = state["research_paper"]
    prompt = f"""You are given the text of a research paper {{PAPER}}.
            Please summarize the paper.
    """"
    generate_response()

    
    draft = f"Draft summary of: {state['research_paper'][:50]}..."


    return {"draft_summary": draft}

def refine_summary(state: SummaryState) -> SummaryState:
    # Simulate refining the summary
    refined = f"Refined: {state['draft_summary']}"
    return {"draft_summary": refined}

def evaluate_summary(state: SummaryState) -> SummaryState:
    # Simulate evaluating the summary
    evaluation = f"Evaluation of: {state['refined_summary']}"
    return {"evaluation": evaluation}


def langchain_driver():
    # Create the graph
    workflow = StateGraph(SummaryState)

    # Add nodes
    workflow.add_node("draft", draft_summary)
    workflow.add_node("refine", refine_summary)
    workflow.add_node("evaluate", evaluate_summary)

    # Add edges
    workflow.set_entry_point("draft")
    workflow.add_edge("draft", "refine")
    workflow.add_edge("refine", "evaluate")
    workflow.add_edge("evaluate", END)

    # Compile the graph
    app = workflow.compile()

    # Run the graph
    result = app.invoke({
        "research_paper": "This is a sample research paper about AI and its applications."
    })

    print(result)


    llm = Fireworks(
        api_key="<KEY>",
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        max_tokens=256)
    llm("Name 3 sports.")
    pass 

def driver():
    pass 


if __name__ == "__main__":
    driver()