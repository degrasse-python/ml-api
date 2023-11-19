from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import joblib
from typing import List

# Load your pre-trained pickled ML model
model = joblib.load('path/to/your/model.pkl')

app = FastAPI()

# Security: OAuth2 password bearer for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User database (for demonstration purposes)
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "testuser@example.com",
        "hashed_password": "fakehashedpassword",
    }
}

# Model for token response
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# Model for user
class User(BaseModel):
    username: str
    email: str
    full_name: str

# Model for input data
class MyData(BaseModel):
    input_1: float
    input_2: float
    # add more input fields as needed

class PredictionResult(BaseModel):
    result: float

# Function to get the current user based on the provided API token
def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    user = fake_users_db.get(token)
    if user is None:
        raise credentials_exception
    return user

# Secure endpoint that requires authentication
@app.post("/predict", response_model=PredictionResult)
async def predict(data: MyData, current_user: User = Depends(get_current_user)):
    try:
        # Validate input data
        if data.input_1 < 0 or data.input_2 < 0:
            raise HTTPException(status_code=422, detail="Input values must be non-negative")

        # Convert the input data to the format expected by the ML model
        input_data = [[data.input_1, data.input_2]]  # Add more inputs as needed

        # Make predictions using the loaded ML model
        predictions = model.predict(input_data)

        # Process the predictions or post-process the results as needed
        result = predictions[0]  # Assuming a single prediction output
        # Add your post-processing logic here

        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Run FastAPI using uvicorn with SSL
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
