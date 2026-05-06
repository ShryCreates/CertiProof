"""
Quick test script for CertValidator API
Run: python test_api.py
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing /health...")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}\n")

def test_register():
    print("Testing /api/v1/auth/register...")
    r = requests.post(f"{BASE_URL}/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "password123",
        "full_name": "Test User"
    })
    print(f"Status: {r.status_code}")
    if r.status_code == 201:
        data = r.json()
        print(f"Token: {data['access_token'][:50]}...")
        print(f"User: {data['user']}\n")
        return data['access_token']
    else:
        print(f"Error: {r.json()}\n")
        return None

def test_login():
    print("Testing /api/v1/auth/login...")
    r = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "password123"
    })
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Token: {data['access_token'][:50]}...")
        return data['access_token']
    else:
        print(f"Error: {r.json()}\n")
        return None

def test_institutions():
    print("Testing /api/v1/institutions...")
    r = requests.get(f"{BASE_URL}/api/v1/institutions?q=IIT")
    print(f"Status: {r.status_code}")
    print(f"Found {len(r.json())} institutions")
    for inst in r.json()[:3]:
        print(f"  - {inst['name']} ({inst['short_name']})")
    print()

def test_verify(token):
    print("Testing /api/v1/verify...")
    # Create a dummy file
    dummy_file = Path("test_cert.pdf")
    dummy_file.write_bytes(b"%PDF-1.4\nDummy certificate for testing")
    
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("test_cert.pdf", open(dummy_file, "rb"), "application/pdf")}
    
    r = requests.post(f"{BASE_URL}/api/v1/verify", headers=headers, files=files)
    print(f"Status: {r.status_code}")
    if r.status_code == 201:
        data = r.json()
        print(f"Verdict: {data['verdict']}")
        print(f"Trust Score: {data['trust_score']}")
        print(f"Institution Match: {data['institution_match']}")
        print(f"Processing Time: {data['processing_time_s']}s")
        print(f"ID: {data['id']}\n")
        return data['id']
    else:
        print(f"Error: {r.json()}\n")
        return None
    finally:
        dummy_file.unlink(missing_ok=True)

def test_history(token):
    print("Testing /api/v1/history...")
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{BASE_URL}/api/v1/history", headers=headers)
    print(f"Status: {r.status_code}")
    print(f"History items: {len(r.json())}")
    for item in r.json()[:3]:
        print(f"  - {item['filename']} → {item['verdict']} ({item['trust_score']})")
    print()

if __name__ == "__main__":
    print("=== CertValidator API Test ===\n")
    
    test_health()
    test_institutions()
    
    # Try register (might fail if user exists)
    token = test_register()
    
    # If register fails, try login
    if not token:
        token = test_login()
    
    if token:
        verify_id = test_verify(token)
        test_history(token)
    else:
        print("❌ Could not authenticate")
    
    print("=== Test Complete ===")
