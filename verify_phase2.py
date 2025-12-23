from fastapi.testclient import TestClient
from backend.server import app, jobs
import os
from pathlib import Path

client = TestClient(app)

def test_catalog():
    print("\n--- Testing /api/models ---")
    response = client.get("/api/models")
    if response.status_code == 200:
        data = response.json()
        print("SUCCESS: Got models:")
        print(data.keys())
        if "catalog" in data and "low" in data["catalog"]:
            print("SUCCESS: Catalog structure looks valid")
        else:
            print("FAIL: Catalog structure invalid")
    else:
        print(f"FAIL: Status code {response.status_code}")
        print(response.text)

def test_prompts():
    print("\n--- Testing /api/prompts ---")
    response = client.get("/api/prompts")
    if response.status_code == 200:
        data = response.json()
        print(f"SUCCESS: Got {len(data['prompts'])} prompts")
        for p in data['prompts']:
            print(f" - {p['name']}")
    else:
        print(f"FAIL: Status code {response.status_code}")

def test_async_switch():
    print("\n--- Testing /api/model/set (Async) ---")
    # 1. Start Job
    # Use a model we know is in the catalog
    target_model = "qwen2.5:1.5b" 
    payload = {"model": target_model}
    
    response = client.post("/api/model/set", json=payload)
    if response.status_code == 200:
        data = response.json()
        job_id = data["job_id"]
        print(f"SUCCESS: Job started with ID: {job_id}")
        
        # 2. Check Status immediately
        status_res = client.get(f"/api/model/status/{job_id}")
        if status_res.status_code == 200:
            status_data = status_res.json()
            print(f"SUCCESS: Initial status: {status_data['status']}")
            print(f"Details: {status_data['details']}")
        else:
            print("FAIL: Could not check status")
            
    else:
        print(f"FAIL: Status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    try:
        test_catalog()
        test_prompts()
        test_async_switch()
    except Exception as e:
        print(f"CRITICAL FAIL: {e}")
