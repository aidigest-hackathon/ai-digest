from langchain import hub
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from typing import List, Dict, Any
from pydantic.v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.callbacks import StdOutCallbackHandler
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os
import sys
import requests
from bs4 import BeautifulSoup
from data_parse.parsing_pdfs import workflow

class FetchResearchContentTool(BaseModel):
    paper_url: str = Field(..., title="URL", description="The URL path to fetch the research content from")
    # metadata: Dict[str, Any] = Field(default={}, title="Metadata", description="Metadata to pass to the model")

@tool(args_schema=FetchResearchContentTool)
def fetch_research_content(paper_url: str, 
                        #    metadata: dict
                           ) -> str:
    """
    Fetch the research content from the path
    """

    loader = WebBaseLoader(paper_url)
    research_text = loader.load()
    
    return '\n'.join([i.page_content for i in research_text])

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

def fetch_content_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the agent on the given state
    """
    #### ARUSHI's func

    research_articles = get_paper_info(state['date'])

    responses = []
    urls = [i.get('pdf_link', '') for i in research_articles]
    print(f"Number of articles: {len(urls)}")

    for url in urls:
        print(f"Fetching content from: {url}")
        # response = fetch_research_content(url)
        try:
            response = workflow(url)
        except Exception as e:
            print(f"Failed to fetch content from {url}: {e}")
            response = None
        
        responses.append(response)
    
    state['research_text'] = responses
    return state

if __name__ == "__main__":
    
    load_dotenv()
    
    papers = get_paper_info('2024-08-08')
    print(papers)

    parsed_papers = workflow(papers[0]['pdf_link'])

    print(f"Number of papers: {len(parsed_papers)}")
    print(f"Avg paper length: {sum([len(i) for i in parsed_papers['research_text']])/len(parsed_papers)}")


    # print(f"Number of papers: {len(parsed_papers)}")
    # print(f"Avg paper length: {sum([len(i) for i in parsed_papers['research_text']])/len(parsed_papers)}")

    # pdf_link = "https://arxiv.org/pdf/2408.02545"
    # workflow(pdf_link)
