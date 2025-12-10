# myfastapi.py
# 2025-12-10 FastAPI Introduction
# FastAPI Introduction
# What is FastAPI?
# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

# Key features of FastAPI:
# Fast: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic).
# Fast to code: Increase the speed to develop features by about 200% to 300%.
# Fewer bugs: Reduce about 40% of human (developer) induced errors.
# Intuitive: Great editor support. Completion everywhere. Less time debugging.
# Easy: Designed to be easy to use and learn. Less time reading docs.
# Short: Minimize code duplication. Multiple features from each parameter declaration.
# Robust: Get production-ready code. With automatic interactive documentation.
# Standards-based: Based on (and fully compatible with) the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.
# Automatic interactive API documentation (provided by Swagger UI and ReDoc).

# FastAPI is faster than Flask and Django
# AI built on top of Starlette for the web parts and Pydantic for the data parts.
# Built-in support for asynchronous programming and Python's async and await keywords.
# Dependency Injection: FastAPI has a simple and powerful
# Dependency Injection system.  
# Install FastAPI and Uvicorn
# pip install fastapi pydantic uvicorn

# How to run the FastAPI app
# % uvicorn myfastapi:app --reload
# Open your browser and go to http://127.0.0.1:8000

# What is CRUD?
# Create
# Read
# Update
# Delete

# What are HTTP Requests?
# GET
# POST
# PUT
# DELETE


from fastapi import FastAPI, HTTPException, status, Path, Query
from pydantic import BaseModel
from typing import List, Optional

# Build FastAPI application
app = FastAPI()

# Simulate a user database    
users = {
    1: {"name": "Alice", "website":"hiaiperf.org", "role":"boss", "age": 30},
    2: {"name": "Bob", "website":"hiaiperf.org", "role":"developer", "age": 25},
    3: {"name": "Charlie", "website":"hiaiperf.org", "role":"manager", "age": 35},
    4: {"name": "Diana", "website":"hiaiperf.org", "role":"designer", "age": 28},
}

# Base Pydantic Models
class User(BaseModel):
    name: str
    website: str
    role: str
    age: int

class UpdateUser(BaseModel):
    name: Optional[str] = None
    website: Optional[str] = None
    role: Optional[str] = None
    age: Optional[int] = None


# Endpoint (URL) examples
# hiaiperf.org/login
# hiaiperf.org/account

# Endpoint models
@app.get("/")
def root():
    # return a JSON
    return {"message": "Welcome to the Introduction to FastAPI"}

# Get User
@app.get("/users/{user_id}", response_model=dict)
def get_user(user_id: int = Path(..., description="The ID of the user to get", ge=0, lt=100)):
    if user_id not in users:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return users[user_id]
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user
# Create a user
@app.post("/users/", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_user(user: dict):
    new_id = max(users.keys()) + 1 if users else 1
    users[new_id] = user
    return users[new_id]

# Update a user
@app.put("/users/{user_id}", response_model=dict)
def update_user(user_id: int, user_update: UpdateUser):
    if user_id not in users:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    stored_user_data = users[user_id]
    # dict is deprecated, use model_dump
    update_data = user_update.model_dump(exclude_unset=True)
    stored_user_data.update(update_data)
    users[user_id] = stored_user_data
    return users[user_id]

# Delete a user
@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int):
    if user_id not in users:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    deleted_user = users.pop(user_id)
    return {"message": "User deleted successfully", "deleted_user": deleted_user}

# List users with optional age filter
@app.get("/users/", response_model=List[dict])
def list_users(min_age: Optional[int] = Query(None, description="Minimum age to filter users", ge=0)):
    if min_age is not None:
        filtered_users = [user for user in users.values() if user["age"] >= min_age]
        return filtered_users
    return list(users.values())
