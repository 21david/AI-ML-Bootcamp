import os
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from pydantic_core._pydantic_core import ValidationError
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from openai import APIError 
from requests.exceptions import HTTPError

import uuid

# ============================= API KEY SETUPS =============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing — chatting functionality disabled. Please add it to an .env file. Obtain an OpenAI API key at www.platform.openai.com.")
    st.stop()
if not NEWS_API_KEY:
    st.warning("NEWS_API_KEY missing — news lookup disabled. Please add it in an .env file. Obtain a NewsAPI key at www.newsapi.org.")

try:
    web_search = TavilySearchAPIWrapper()
except ValidationError:
    st.warning("TAVILY_API_KEY missing — web search disabled. Please add it in an .env file. Obtain a Tavily key at www.tavily.com.")
    web_search = None


# ============================= TOOL DEFINITIONS =============================
@tool
def search_web(query: str) -> list[dict] | str:
    """Make a web search with a given query"""
    try:
        results = web_search.results(
            query, 
            max_results=2
        )
        if not results:
            return f"No search results found for query: '{query}'."
        return results

    except Exception as e:
        return f"Search provider error: {str(e)}"

@tool
def get_news(category: str) -> list[dict] | str:
    """Retrieve news articles about the given category"""
    try: 
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "apiKey": NEWS_API_KEY,
                "q": category,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 3,
            },
            timeout=10,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return "Error: Request timed out."
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error ({response.status_code}): {e}"
    except requests.exceptions.RequestException as e:
        return f"Network Error: {e}"

    try:
        data = response.json()
    except (requests.exceptions.JSONDecodeError, ValueError):
        return "Invalid JSON from News API"

    if data.get("status") != "ok" or not data.get("articles"):
        return f"Error retrieving articles: {data}"
    
    return data['articles']


# ============================= STATE DEFINITION =============================
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================= SYSTEM PROMPT =============================
tools = []
system_prompt ="""
You are NewsGenie, an information and news assistant.
If using external tools:
- If the user did not specify a clear topic, DO NOT call any tools. Ask the user to specify a topic.
- Prefer trustworthy and reputable sources for all external information.
- You must filter misinformation as best as you can.
- If they return an error or no relevant results, briefly explain it to the user along with best solutions.
- If a user request contains multiple complex tasks, execute tool calls step-by-step.

You must categorize the type of query the user gives you and follow the corresponding instructions:
• General knowledge or chat that you are confident you know the answer to.
    - Answer directly. Do not call a tool.""" 

if NEWS_API_KEY:
    tools.append(get_news)
    system_prompt += """
• News request.
    - Use the 'get_news' tool to retrieve relevant and up-to-date news and summarize them."""

if web_search:
    tools.append(search_web)
    system_prompt += """
• Facts that may be recent or updated or that you don't already know.
    - Use the 'search_web' tool to retrieve relevant information and summarize it.""" 


# ============================= LLM SETUP =============================
llm = ChatOpenAI(
    model='gpt-4o-mini',
    base_url= "https://openai.vocareum.com/v1",  # from Vocareum, not api.openai.com
).bind_tools(tools)


# ============================= GRAPH NODES =============================
def agent(state: ConversationState) -> ConversationState:
    system_message = SystemMessage(content=system_prompt)

    response = llm.invoke(
        [system_message] + state['messages']
    )

    if response.tool_calls:
        print('TOOL USED: ', response.tool_calls, '\n')
    return {'messages': [response]}


# ============================= BUILD GRAPH =============================
graph = StateGraph(ConversationState)

graph.add_node('agent', agent)
graph.add_node('tools', ToolNode(tools=tools, messages_key="messages"))

graph.add_edge(START, 'agent')
graph.add_edge('tools', 'agent')
graph.add_conditional_edges("agent", tools_condition)


# ============================= STREAMLIT ============================= 
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.set_page_config(
    page_title="NewsGenie",
    page_icon="📰",
    layout="centered",
)

st.title("NewsGenie")
st.caption("AI-powered information and real-time news assistant")

with SqliteSaver.from_conn_string('chat_history.sqlite') as memory:
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    
    # Render existing conversation
    state = app.get_state(config)
    if state and state.values:
        for message in state.values.get('messages', []):
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, AIMessage) and message.content:
                st.chat_message("assistant").write(message.content)

    def send_prompt(user_prompt: str) -> None:
        st.chat_message("user").write(user_prompt)

        try:
            app.invoke(
                {"messages": [HumanMessage(content=user_prompt)]},
                config=config
            )
            st.rerun()
        except (APIError, HTTPError) as e:
            st.error(f"API Error: {e}")
            st.stop()
            
    st.write("**Quick prompts:**")
    col1, col2, col3 = st.columns(3, gap="small") 

    if col1.button("🤖 Technology", use_container_width=True):
        send_prompt("Get me recent news about technology.")
        
    if col2.button("📈 Finance", use_container_width=True):
        send_prompt("Get me recent news about finance.") 
        
    if col3.button("⚽ Sports", use_container_width=True):
        send_prompt("Get me recent news about sports.")

    if user_prompt := st.chat_input("Ask a question or request news"):
        send_prompt(user_prompt)
