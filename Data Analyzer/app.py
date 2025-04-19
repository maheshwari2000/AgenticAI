import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from typing import Optional

# Store the CSV globally so all tools can access it
csv_data: Optional[pd.DataFrame] = None

from tools.data_tools import read_csv_tool, check_data_types_tool, check_duplicates_tool, clean_data_tool, missing_data_tool, stat_data_tool, detect_outliers_tool, correlation_matrix_tool


tools = [
    # read_csv_tool, summary_data_tool, missing_data_tool
    read_csv_tool, check_data_types_tool, check_duplicates_tool, clean_data_tool, missing_data_tool, 
    stat_data_tool, detect_outliers_tool, correlation_matrix_tool
]

with open('prompts.txt', 'r') as file:
    react_prompt_template = file.read()

from langchain.agents import initialize_agent
from langchain_groq import ChatGroq
from langchain.agents import AgentType

# Initialize Groq LLM (replace with API key)
llm = ChatGroq(model_name="llama3-8b-8192")

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create an agent that can call our tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Shows intermediate steps
    handle_parsing_errors=True,
    system_message=react_prompt_template,
    memory=memory
)

csv_path = "upload/Mobiles_Dataset_2025.csv"  # Replace with actual CSV path
input_prompt = f'''Analyze the CSV file at {csv_path}. 
Provide a very detailed final analysis of the dataset with all available data and numbers and formatting. 
Use all the tools to analyze, find missing values, stats, correlation matrix between columns, outliers and duplicate data.'''
response = agent.run(input_prompt)
print(response)

# import gradio as gr

# # Define the Gradio interface
# def analyze_csv(file):
#     # Save the uploaded file path
#     csv_path = file.name
#     input_prompt = f'''Analyze the CSV file at {csv_path}. 
#     Provide a very detailed final analysis of the dataset with all available data and numbers and formatting. 
#     Use all the tools to analyze, find missing values, stats, correlation matrix between columns, outliers and duplicate data.'''
#     response = agent.run(input_prompt)
#     return response

# interface = gr.Interface(
#     fn=analyze_csv,  # Function to process CSV
#     inputs=gr.File(label="Upload CSV File"),  # File input
#     outputs=gr.Textbox(label="Analysis Report"),  # Output for agent's response
#     live=True,  # Optionally, updates as the file is uploaded
#     title="CSV Data Analysis Agent",  # UI Title
#     description="Upload a CSV file, and the agent will analyze it for missing data and summary statistics.",  # Description
# )

# # Launch the Gradio app
# interface.launch()