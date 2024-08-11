'''
Need to set up the langchain
Need to pull data from bucket
Need to upload data to bucket


Front-end: need to give the data to display
'''
from dotenv import load_dotenv
env = load_dotenv()
if not env:
    print("Failed to load .env file")
from langchain_fireworks import Fireworks 
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages 

from typing import TypedDict, Annotated, Optional
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os 
import openai
from pydantic import BaseModel, Field
from typing import Optional
import json 
from bs4 import BeautifulSoup
from data_parse.parsing_pdfs import workflow
import requests

load_dotenv()

# Define the state
class SummaryState(TypedDict):
    research_paper: str
    draft_summary: str
    date: str
    title: str
    beginner_result: Optional[str]
    intermediate_result: Optional[str]
    advanced_result: Optional[str]
    image_path: Optional[str]
    link: Optional[str]
    keep_refining: bool 
    entities: list[str] 
    draft_version: int 

def get_paper_info(date_str)->list:
    # Construct the URL based on the date
    base_url = 'https://huggingface.co/papers'
    full_url = f'{base_url}?date={date_str}'

    # Send a GET request to the URL
    response = requests.get(full_url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve data: {response.status_code}")
        return []

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')


    # Find all paper entries
    papers_info = []
    for paper in soup.find_all('h3', class_='mb-1 text-lg/6 font-semibold hover:underline peer-hover:underline 2xl:text-[1.2rem]/6'):
        # Extract PDF link
        pdf_link = None
        title = None
        pdf_tag = paper.find('a', href=True)

        if pdf_tag:
            title = pdf_tag.get_text(strip=True)
            pdf_link = pdf_tag['href']
            # Construct the full URL if necessary
            if not pdf_link.startswith('http'):
                pdf_link = f"https://arxiv.org/pdf{pdf_link}"


        # Append paper info to the list
        papers_info.append({
            'pdf_link': pdf_link,
            'title': title
        })

    return papers_info

class Result(BaseModel):
    missing_entities: list[str]
    summary: str 

def build_state(date:str):
    # 'YYYY-MM-DD'

    state = {}
    
    research_papers = get_paper_info(date) # returns a list of dict with pdf_link and title

#     "research_paper": """Abstract
# The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

# ...
# """,
#         "draft_summary": "",  # Initialize with an empty string
#         "keep_refining": True,  # Initialize as should continue 
#         "entities": [],  # Initialize with an empty list
        # "draft_version"
    
    for research_paper in research_papers:
        parsed_paper = workflow(research_paper['pdf_link']) # returns a list of strings

        state['research_paper'] = parsed_paper
        state['draft_summary'] = ""
        state['keep_refining'] = True
        state['entities'] = []
        state['draft_version'] = 0

    return state

class GPTSummarizer:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.getenv("FIREWORKS_API_KEY"),
        )

    def generate(self, state: SummaryState):
        '''
        Generates a response, returns a SummaryState
        '''
        # first generation
        draft_version = state["draft_version"]
        research_paper = state['research_paper']
        previous_summary = state['draft_summary']
        draft_version = state['draft_version']
        current_entities = state["entities"]
        if draft_version == 0:
            first_summary_prompt = f'''Given the full text of a computer science research paper, we want to generate a summary that is concise yet dense in information.
This will proceed with increasingly helpful and entity-dense summaries of the article.
Step 1. Identify 2-3 informative entities from the paper which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the new "Missing Entities". A Missing Entity is:
  - Relevant: to the main story.
  - Specific: descriptive yet concise (5 words or fewer).
  - Novel: not in the previous summary.
  - Faithful: present in the Article.
  - Anywhere: located anywhere in the Article.
We are dealing with the first version of the summary, which should be 5-7 bulletpoint or around ~80 words. Feel free to start with highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
Missing entities can appear anywhere in the new summary. Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
Your final response should be in well-formatted JSON whose keys are "missing_entities" (list) and "summary" (string). There should be no further text or explanation in the response.
The input will be the following format: Input: {{PAPER}}
For example,
_Output_
```json
{{
  "missing_entities": ["entity1", "entity2"],
  "summary": "The authors describe a method of synthesizing data using an external data corpus for improving the diversity. This works well enough, but has drawbacks due to the complexity of the pipeline. The key innovations include a novel sub-sampling procedure as well as a new method for data augmentation."
}}
Now it is your turn, please generate relevant entities and a denser summary.

INPUT: {research_paper}
_Output_'''

            
            history = [{"role": "user", "content": first_summary_prompt}]

        # refiner
        else:
            refine_summary_prompt = f'''Given the full text of a computer science research paper, we want to generate a summary that is concise yet dense in information.
This will proceed with increasingly helpful and entity-dense summaries of the article.
Step 1. Identify 2-3 informative entities from the paper which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the new "Missing Entities". A Missing Entity is:
  - Relevant: to the main story.
  - Specific: descriptive yet concise (5 words or fewer).
  - Novel: not in the previous summary.
  - Faithful: present in the Article.
  - Anywhere: located anywhere in the Article.
We are dealing with the {draft_version} version of the summary, which should be 5-7 sentences matching the length of the original summary. Make every word count. Re-write the previous summary to improve flow and make space for additional entities. Make space with fusion, compression, and removal of uninformative phrases like "the article discusses". The summaries should provide commentary and an interpretation of the results, such as the practicality of implementation. Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
Current Entities: {current_entities}
Previous Summary: {previous_summary}
Research Paper: {research_paper}
Your final response should be in well-formatted JSON whose keys are "missing_entities" (list) and "summary" (string). There should be no further text or explanation in the response.
For example,
_Output_
```json
{{
  "missing_entities": ["synthesize", "diversity"],
  "summary": "The authors describe a method of synthesizing data using an external data corpus for improving the diversity. This works well enough, but has drawbacks due to the complexity of the pipeline. The key innovations include a novel sub-sampling procedure as well as a new method for data augmentation."
}}
Now it is your turn, please generate relevant entities and a denser summary.
_Output_'''
            history = [{
                "role": "user",
                "content": refine_summary_prompt,
            }]
        
        chat_completion = self.client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            response_format={"type": "json_object", "schema": Result.model_json_schema()},
            messages=history
        )
        # TODO: Change to JSON
        json_string = chat_completion.choices[0].message.content
        json_object = json.loads(json_string)
        new_entities = json_object["missing_entities"]
        summary = json_object['summary']
        current_entities = [*current_entities, *new_entities]
        state = {"draft_summary": summary, "draft_version": draft_version + 1, "research_paper": research_paper, "entities": current_entities}
    
        return state 




# Define the nodes
def draft_summary(state: SummaryState) -> SummaryState:
    # Simulate drafting a summary
    summarizer = GPTSummarizer()
    new_state = summarizer.generate(state)
    return new_state

def refine_summary(state: SummaryState) -> SummaryState:
    # Simulate refining the summary
    summarizer = GPTSummarizer()
    new_state = summarizer.generate(state)
    return new_state

def evaluate_summary(state: SummaryState) -> SummaryState:
    # Adds a boolean to the field keep_refining in SummaryState. True means keep generating, False means good enough 
    keep_refining = False  
    #evaluation = f"Evaluation of: {state['refined_summary']}"
    new_state = state.copy()
    new_state['keep_refining'] = keep_refining   
    breakpoint() 
    return new_state


def should_continue(state: SummaryState):
    breakpoint()
    is_continue = state['keep_refining']
    if is_continue:
        return "refine"
    else:
        return END



def get_complex_summary():
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
    workflow.add_conditional_edges("evaluate", should_continue)
    
    #workflow.add_edge("evaluate", END) #TODO: maybe optional

    # Compile the graph
    app = workflow.compile()

    state = build_state('2024-08-08')

    result = app.invoke(state)
        # Run the graph
#     result = app.invoke({
#         "research_paper": """Abstract
# The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

# ...
# """,
#         "draft_summary": "",  # Initialize with an empty string
#         "keep_refining": True,  # Initialize as should continue 
#         "entities": [],  # Initialize with an empty list
#         "draft_version": 0  # Initialize as 0
#     })
    print(result)
    return result

class StyleGen:
    def __init__(self, draft):
        self.draft = draft 
        self.client = openai.OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.getenv("FIREWORKS_API_KEY"),
        )  


    def get_styled(self, style_prompt):        
        history = [{
                "role": "user",
                "content": style_prompt,
            }]
        
        chat_completion = self.client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            messages=history
        )

        return chat_completion.choices[0].message.content


def driver():
    summary_state = get_complex_summary()
    breakpoint()
    complex_summary = summary_state["draft_summary"]

    styler = StyleGen(complex_summary)
    beginner_prompt = f"""Given a technical summary of a computer science research paper, please output a beginner-friendly summary with no additional explanation or starting text, just the final summary itself.
    Your input is summary: {{Summary}}

    Summary: {complex_summary}"""

    intermediate_prompt = f"""Given a technical summary of a computer science research paper, please output an intermediately technical summary with no additional explanation or starting text, just the final summary itself.
    Your input is summary: {{Summary}}

    Summary: {complex_summary}"""

    beginning_summary = styler.get_styled(beginner_prompt)
    intermediate_summary = styler.get_styled(intermediate_prompt)

    new_state = summary_state.copy()
    new_state["beginner_result"] = beginning_summary
    new_state["intermediate_result"] = intermediate_summary
    new_state["advanced_result"] = complex_summary
    return new_state





if __name__ == "__main__":
    driver()