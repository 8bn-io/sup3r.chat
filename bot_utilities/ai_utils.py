import os
import logging
from urllib.parse import quote
from dotenv import find_dotenv, load_dotenv
from bs4 import BeautifulSoup
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import LLMMathChain, ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from bot_utilities.config_loader import config, load_personas
import openai
from bot_utilities.config_loader import config, load_current_language, load_personas

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables and configurations
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL_CHAT = config['GPT_MODEL_CHAT']
GPT_MODEL_BASE = config['GPT_MODEL_BASE']
MAX_RESPONSE_LENGTH = config["MAX_RESPONSE_LENGTH"]
SELECTED_TEMPERATURE = config["CHAT_TEMPERATURE"]
SELECTED_PERSONA = config['PERSONA']
PERSONA_INSTRUCTION = load_personas().get(SELECTED_PERSONA)
PERSIST = config["PERSIST"]
MAX_CHAT_HISTORY_LENGTH = config["MAX_CHAT_HISTORY_LENGTH"]
MAX_MSG_HISTORY_LENGTH = config["MAX_MSG_HISTORY_LENGTH"]
K_DOCS = config["K_DOCS"]
LAMBDA_MULT = config["LAMBDA_MULT"]


try:
    # Exepction handling
    if MAX_MSG_HISTORY_LENGTH < 0:
        error_message = "Maximum message history length should be a non-negative number"
        logging.info(error_message)
        raise ValueError(error_message)

    if MAX_CHAT_HISTORY_LENGTH <= MAX_MSG_HISTORY_LENGTH:
        error_message = "Maximum chat history length should be greater than maximum message history length"
        logging.info(error_message)
        raise ValueError(error_message)
except Exception as e:
    logging.error(f"An error occurred while setting message history or chat history length: {str(e)}")

    # Set default values in case of an error
    MAX_MSG_HISTORY_LENGTH = 4
    MAX_CHAT_HISTORY_LENGTH = 2




print(f"Persist : {PERSIST}")

class ChatbotManager:
    """ Manages the chatbot agents and interactions. """

    def __init__(self):
        """ Initializes the chatbot manager with default settings. """
        self.current_language = load_current_language()
        self.retrieval_index = None
        self.agents = {}
        self.message_history = {}

    def create_retrieval_index(self, persist=PERSIST):
        """ Creates a retrieval index. """
        try:
            if persist and os.path.exists("persist"):
                logging.info("Loading existing index from 'persist' directory.")
                vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
                self.retrieval_index = VectorStoreIndexWrapper(vectorstore=vectorstore)
            else:
                loader = DirectoryLoader("data/")
                index_creator = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"} if persist else {})
                self.retrieval_index = index_creator.from_loaders([loader])
            logging.info(f"Created new index with {self.retrieval_index.vectorstore} vectors.")
        except Exception as e:
            logging.error("Error creating retrieval index", exc_info=True)
            raise

    def create_agent(self, id, user_name, Chatbot, instructions):
        """ Creates an agent for a given user. """
        system_message = SystemMessage(content=instructions)
        logging.info(f"Creating agent for user {user_name}.")
        logging.info(f"System Message: {str(system_message)[:30]} ...")
        logging.info(f"Conversation history: {self.message_history.get(id, [])[:50]} ...")

        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": system_message,
        }

        memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, ai_prefix=SELECTED_PERSONA, human_prefix=user_name)
        print(f"LLM settings are: a) Temp: {SELECTED_TEMPERATURE} and b) {GPT_MODEL_CHAT} \n")

        try:
            agent = initialize_agent(
                tools = [],
                llm=self.get_llm(), 
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
                agent_kwargs=agent_kwargs,
                memory=memory
            )
            # To Do: this is wrong. 
            self.agents[id] = agent
            hashed = abs(hash(str(agent)))
            hash_5digits = str(hashed % 100000)
            logging.info(f"Agent {hash_5digits} created for user {user_name}.")
        except Exception as e:
            logging.error("Error creating agent", exc_info=True)
            print(f"Error creating agent: {e}")
            raise

    def get_llm(self):
        """ Returns an instance of ChatOpenAI. """
        return ChatOpenAI(temperature=SELECTED_TEMPERATURE, model=GPT_MODEL_CHAT)
    
    def get_retriever(self):
        """ Returns an instance of VectorStoreIndexWrapper. 
        Setting for Retriever: k = 6 documents with lamda_mult = 0,25 for higher diversity
        Those settings are useful if your dataset has many similar documents
        """
        return self.retrieval_index.vectorstore.as_retriever(search_kwargs={'k': K_DOCS, 'lambda_mult': LAMBDA_MULT})

    def generate_response(self, instructions, user_input):   
        """ Generates a response based on user input. """
        if self.retrieval_index is None:
            self.create_retrieval_index(persist=PERSIST)

        user_channel_id = user_input.get("user_channel_id")    
        message = user_input.get("message")
        print(f"\n\n #### Message is: {message[:30]} ### \n\n")
        user_name = user_input.get("user_name")
        Chatbot = user_input.get("Chatbot")

        if user_channel_id not in self.agents:
            self.create_agent(user_channel_id, user_name, Chatbot, instructions)
            self.message_history.setdefault(user_channel_id, [])

        agent = self.agents[user_channel_id]

        
        #
        # template = (
        # "Combine the chat history and follow up question into "
        # "a standalone question. Chat History: {chat_history}"
        # "Follow up question: {question}"
        # )
        # To Do: I need to insert the persona here.
        

        chat_history = self.message_history.get(user_channel_id, [])[-MAX_CHAT_HISTORY_LENGTH:0]
        ''' Get user and channel specific recent chat history from the long term message history
        '''
        print("\n#tik#\n") # checking if i get here
        print(len(chat_history))
        print("\n#tok#\n") # checking if i get here
        logging.info(f"Chat history for user {user_channel_id}: {chat_history[:50]} ...")
        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.get_llm(),
                retriever=self.get_retriever() 
            )

            chat_message = PERSONA_INSTRUCTION + " " + message
            # Add persona instruction to the prompt for GPT

            result = chain({"question": chat_message, "chat_history": chat_history}) ## backup
            response = result['answer']
            self.message_history[user_channel_id].append((message, response))
            self.message_history[user_channel_id] = self.message_history[user_channel_id][-MAX_MSG_HISTORY_LENGTH:]
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            raise

        '''            
        from langchain.chains import (
                StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
            )
        from langchain_core.prompts import PromptTemplate ## I'm using from langchain.prompts import PromptTemplate
        from langchain.llms import OpenAI

            combine_docs_chain = StuffDocumentsChain(...)
            vectorstore = ...
            retriever = vectorstore.as_retriever()

            # This controls how the standalone question is generated.
            # Should take `chat_history` and `question` as input variables.
            template = (
                "Combine the chat history and follow up question into "
                "a standalone question. Chat History: {chat_history}"
                "Follow up question: {question}"
            )
            prompt = PromptTemplate.from_template(template)
            llm = OpenAI()
            question_generator_chain = LLMChain(llm=llm, prompt=prompt)
            chain = ConversationalRetrievalChain(
                combine_docs_chain=combine_docs_chain,
                retriever=retriever,
                question_generator=question_generator_chain,
        '''

# Usage example
#chatbot_manager = ChatbotManager()
#response = chatbot_manager.generate_response(instructions="Some instructions", user_input={"id": 1, "message": "Hello", "user_name": "John", "Chatbot": "AI"})
#print(response)