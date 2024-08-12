import os
import re
import openai
import sys
import requests
import fitz
from langchain.document_loaders import PyPDFLoader

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

def parse_pdf(pdf_path:str):
    # url = f"https://arxiv.org/pdf/{arxiv_no}"
    # response = requests.get(url)
    # pdf_file = "paper.pdf"

    # with open(pdf_file, 'wb') as f:
    #     f.write(response.content)


    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = ""
    for page in pages:
        text += page.page_content

    pdf_document = fitz.open(pdf_file)

    # os.makedirs(arxiv_no, exist_ok=True)

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)
        
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{arxiv_no}/{page_number+1}_{image_index+1}.png"

            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
            print(f"Extracted: {image_filename}")
    
    return text


def extract_sections_by_regex(text):
    # Regex pattern to match the start of a section
    pattern = r"((?m)^(?:\d|[IVXL]+\.)\s[A-Za-z-:]+\s*(?:[A-Za-z-]+\s*){0,3}\n)"

    # Find all matches for the section headers
    headers = re.finditer(pattern, text)

    # Extract the sections
    sections = {}
    last_header = None
    last_position = None

    for header in headers:
        header_text = header.group(0).strip()
        start_position = header.start()

        if last_header:
            # Extract the text between the last header and the current header
            section_text = text[last_position:start_position].strip()
            sections[last_header] = section_text

        # Update the last header and position
        last_header = header_text
        last_position = start_position

    # Capture the last section after the final header
    if last_header:
        sections[last_header] = text[last_position:].strip()

    return sections

def remove_text_after_references(text):
    # Regex pattern to delete everything after "References"
    pattern = r"(?i)References[\s\S]*"
    
    # Replace the "References" section and everything after it with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text


def extract_sections_around_abstract(text):
    # Regex pattern to match the "Abstract" section
    pattern = r"(?i)(.*?)(^abstract\s*)(.*?)(^\d+\s+.*$)"

    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)

    if match:
        before_abstract = match.group(1).strip()  # Text before "Abstract"
        abstract = match.group(3).strip()         # Text within the "Abstract"
        after_abstract = match.group(4).strip()   # Text after the "Abstract"

        return before_abstract, abstract, after_abstract
    else:
        return None, None, None


def workflow(pdf_link):
    pdf_no = pdf_link.split("/")[-1]
    # for pdf_no in ["2408.02442", "2408.02479", "2408.04632", "2408.04631", "2408.04628", "2401.07061"]:
    # # for pdf_no in ["2401.07061"]:
    text = parse_pdf(pdf_no)
    before_abstract, abstract, after_abstract = extract_sections_around_abstract(text)
    cleaned_text = remove_text_after_references(after_abstract)
    sections = extract_sections_by_regex(cleaned_text)
    if len(sections) < 4:
        print(f"Error: {pdf_no} has less than 4 sections, it only has {len(sections)} sections")
        # return
    with open(f"{pdf_no}/abstract.txt", "w") as f:
        f.write(abstract)

    final_text = f"--- ABSTRACT ---\n{abstract}\n"    
    for raw_sec_name, content in sections.items():
        section = classify_section(raw_sec_name, content)

        final_text += f"--- {raw_sec_name.upper()} ---\n"
        final_text += content
        final_text += "\n"
        with open(f"{pdf_no}/{section.lower()}.txt", "w") as f:
            f.write(content)

    with open(f"{pdf_no}_final_text.txt", "w") as f:
        f.write(final_text)

def classify_section(sec_name, content):
    # assign each section to a category of either introduction, methods, results, or conclusion.
    tokens = sec_name.split()
    if "introduction" in tokens or 'related' in tokens:
        return "introduction"
    elif "conclusion" in tokens or 'discussion' in tokens:
        return "conclusion"
    else:
        for tag in ['experiment', 'setup', 'method', 'model', 'approach']:
            if tag in sec_name:
                return "methods"

    return "results"

if __name__ == "__main__":
    pdf_link = "https://arxiv.org/pdf/2408.02545"
    workflow(pdf_link)


