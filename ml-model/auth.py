from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from typing import List

app = FastAPI()

# Sample users database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    }
}

# Secret key to sign JWT tokens
SECRET_KEY = "c7c4d31520043e07f8452f9c6b1c37e26db3459fcd230eafaf892df041ada338"
ALGORITHM = "HS256"

# OAuth2 for password flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Function to create a JWT token
def create_jwt_token(data: dict):
    to_encode = data.copy()
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Function to decode a JWT token
def decode_jwt_token(token: str):
    credentials_exception = HTTPException(
        status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise credentials_exception

# Dependency for authentication
def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials = decode_jwt_token(token)
    username: str = credentials.get("sub")
    if username is None:
        raise credentials_exception
    return credentials

# Sample endpoint with authentication and authorization
@app.get("/users/me", response_model=dict)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user

# Sample endpoint with input validation
@app.post("/items/")
async def create_item(item: dict):
    return item
