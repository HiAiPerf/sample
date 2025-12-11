# Examples of FastAPI, LangChain, and LangGraph

==============================================================================
# Example 1: Basic FastAPI example
# Source code myfastapi.py

Install:

$ conda create -n fastapi311 python=3.11

$ conda activate fastapi311

$ pip install -qU fastapi pydantic uvicorn

How to use it:

python myfastapi.py



==============================================================================
# Examle 2: FastAPI and LangGraph with Tool

# Source code: graph.py server.py, ui.html

Install:

$ conda create -n langgraph311 python=3.11

$ conda activate langgraph311

$ pip install -qU fastapi pydantic uvicorn

$ pip install -qU python-dotenv os

$ pip install -qU langgraph langchain-core langsmith langchain_anthropic

$ pip install -qU langchain langchain-openai langchain-anthropic langchain-ollama

How to run:

uvicorn server:app --reload --port 8000


How to test:

curl 'http://127.0.0.1:8000/chat?query=What+is+the+weather+in+Tokyo'  -H 'accept: application/json'

curl 'http://127.0.0.1:8000/chat?query=What+is+the+weather+in+London'  -H 'accept: application/json'

curl 'http://127.0.0.1:8000/chat?query=What+is+the+weather+in+Toronto'  -H 'accept: application/json'
