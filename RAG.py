import torch
import fitz
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import textwrap
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import os
import openai
from openai import OpenAI
import gradio as gr
import json

PDF_PATH= "CCLSpring2025.pdf"

def reformat_text(text: str):
  text = text.replace("\n", " ")
  return text

# load the information in the pdf into a list of dictionaries so we can
# easily store them into a dataframe
def chunk_pdf(doc,chunk_size=6):
  text_per_page = []
  # go to each page
  for page_number, page in enumerate(doc):
    sentence = ''
    accumulated_text=''
    # look at each line in the document
    for i,text in enumerate(page.get_text("text").split("\n")):

      # get rid of any header text
      if text.upper() == text:
        text=text.replace(text, " ")

      # once we've reached our chunk size, add that chunk to the dictionary which we will convert to a dataframe (this step is just so that we can see how the chunks look more easily)
      accumulated_text += text
      if i > 0 and i % chunk_size == 0:
        sentence += accumulated_text
        sentence = reformat_text(sentence)
        text_per_page.append({"Text": sentence,
                          "Page_#": page_number,

        })
        accumulated_text = ''
        sentence = ''

  return text_per_page

# See how the datatable looks after we chunk our document
doc = fitz.open(PDF_PATH)
pd.DataFrame(chunk_pdf(doc))

#open the document, chunk the text, and store each chunk as an entry in the text_chunks list
doc = fitz.open(PDF_PATH)
chunk_size=5
chunks = chunk_pdf(doc,chunk_size=chunk_size)
text_chunks = []
for chunk in chunks:
  text_chunks.append(chunk['Text'])
print(f"Number of chunks: {len(text_chunks)}")
print(f"Text chunk 1 is {text_chunks[1]}")

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API base and model name from environment variables
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai_api_model_name = os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o")

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please add your OpenAI API key to the .env file.")

# Configure OpenAI API
# Configure OpenAI API
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

def query_openai_gradio(activity_name: str):
    """
    Function to query OpenAI for topic ideas.
    """
    try:
        # Construct the system prompt for generating topic ideas
        system_prompt = f"""You are a creative assistant. Generate a list of topic ideas for Parsons problems.
The topics should be relevant to the activity: "{activity_name}".
Provide 5-10 diverse and interesting topics that could be discussed in this activity.
The output must be a valid JSON array of strings, like this:
["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"]"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model=openai_api_model_name,
            messages=[{"role": "system", "content": system_prompt}],
            response_format={"type": "json_object"}
        )

        # Extract the AI-generated content
        ai_response = response.choices[0].message.content
        topics = json.loads(ai_response)

        return topics

    except Exception as e:
        return f"Error generating topics: {str(e)}"

# Create Gradio interface
interface = gr.Interface(
    fn=query_openai_gradio,
    inputs=gr.Textbox(label="Activity Name", placeholder="Enter the activity name"),
    outputs=gr.JSON(label="Generated Topics"),
    title="OpenAI Topic Generator",
    description="Enter an activity name to generate a list of topic ideas using OpenAI."
)

if __name__ == "__main__":
    interface.launch()