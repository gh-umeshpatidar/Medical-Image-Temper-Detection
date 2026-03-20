# Explain this code and optimize it

from fastapi import Header, HTTPException

API_KEYS = ["abc123", "research_key_001"]

def verify_api_key(authorization: str = Header(...)):
    if authorization.replace("Bearer ", "") not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
