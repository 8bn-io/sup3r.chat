import aiohttp
import io
import time
import random
import os
import json
import requests
import openai

# Importing from bs4 and pydantic
from bs4 import BeautifulSoup
from pydantic import Field
from urllib.parse import quote
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from bot_utilities.config_loader import load_current_language, config

# Importing from langchain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import LLMMathChain, ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter



load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

current_language = load_current_language()


# Global variable to keep track of the retrieval index
retrieval_index = None

def create_retrieval_index(persist=True):
    """
    Create or load the retrieval index.
    """
    global retrieval_index
    if persist and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        retrieval_index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data/")
        if persist:
            retrieval_index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            retrieval_index = VectorstoreIndexCreator().from_loaders([loader])



## To Do - Does not work
## Add embeddings here
def knowledge_retrieval(query):    
    # Define the data to be sent in the request
    data = {
        "params":{
            "query":query
        },
        "project": "feda14180b9d-4ba2-9b3c-6c721dfe8f63"
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post("https://api-1e3042.stack.tryrelevance.com/latest/studios/6eba417b-f592-49fc-968d-6b63702995e3/trigger_limited", data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        return response.json()["output"]["answer"]
    else:
        print(f"HTTP request failed with status code {response.status_code}") 

## create own function to summarize
def summary(content):
    llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    summary_chain = load_summarize_chain(
        llm=llm, 
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = True
    )

    output = summary_chain.run(input_documents=docs,)

    return output


def scrape_website(url: str):
    #scrape website, and also will summarize the content based on objective if the content is too large
    #objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url        
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post("https://chrome.browserless.io/content?token=2db344e9-a08a-4179-8f48-195a2f7ea6ee", headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)
        if len(text) > 10000:
            output = summary(text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")     


def search(query):
    """
    Asynchronously searches for a prompt and returns the search results as a blob.

    Args:
        prompt (str): The prompt to search for.

    Returns:
        str: The search results as a blob.

    Raises:
        None
    """

    endpoint = "https://ddg-api.herokuapp.com/search"
    params = {
        'query': query,  # Replace with your search query
        'limit': 5  # Replace with your desired limit
    }
    
    # Make the GET request
    response = requests.get(endpoint, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        results = response.json()
        return results
    else:
        return (f"Didn't get any results")



def research(query):
    system_message = SystemMessage(
        content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You will always searching for internal knowledge base first to see if there are any relevant information
            2/ If the internal knowledge doesnt have good result, then you can go search online
            3/ While search online:
                a/ You will try to collect as many useful details as possible
                b/ If there are url of relevant links & articles, you will scrape it to gather more information
                c/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
    )

    agent_kwargs = {
        "system_message": system_message,
    }

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [        
        Tool(
            name="Knowledge_retrieval",
            func=knowledge_retrieval,
            description="Use this to get our internal knowledge base data for curated information, always use this first before searching online"
        ),      
        Tool(
            name = "Google_search",
            func = search,
            description = "Always use this to answer questions about current events, data, or terms that you don't really understand. You should ask targeted questions"
        ),          
        Tool(
            name = "Scrape_website",
            func = scrape_website,
            description = "Use this to load content from a website url"
        ),   
    ]

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        agent_kwargs=agent_kwargs,
    )

    results = agent.run(query)

    return results
 
async def fetch_models():
    return openai.Model.list()
    
agents = {}

def create_agent(id, user_name, ai_name, instructions):
    system_message = SystemMessage(
        content=instructions
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }

    memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, ai_prefix=ai_name, user_prefix=user_name)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    tools = [                     
        Tool(
            name = "research",
            func = research,
            description = "Always use this to answer questions about current events, data, or terms that you don't really understand. You should ask targeted questions"
        ),           
        Tool(
            name = "Scrape_website",
            func = scrape_website,
            description = "Use this to load content from a website url"
        ),   
    ]    

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory
    )

    agents[id] = agent

    print("Agent " + str(agent) + " created")
    
    return agent

def generate_response(instructions, user_input):   
    message_history = {} # should be defined as global variable in main.py
    global retrieval_index

    if retrieval_index is None:
        create_retrieval_index(persist=True)

    id = user_input["id"]    
    message = user_input["message"]

    if id not in agents:
        user_name = user_input["user_name"]
        ai_name = user_input["ai_name"]
        agent = create_agent(id, user_name, ai_name, instructions)
        # Initialize message_history for a new id
        message_history.setdefault(id, [])
    else:
        agent = agents[id]

    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=retrieval_index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )
    
    # Generate response and append to chat history
    chat_history = message_history.get(id, [])  # Get chat history or an empty list
    result = chain({"question": message, "chat_history": chat_history})
    response = result['answer']
    message_history.setdefault(id, []).append((message, response))

    return response