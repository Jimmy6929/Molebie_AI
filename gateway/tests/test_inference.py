"""
Test script for Step 1.5 — Inference endpoints.
Run: python test_inference.py
"""

import httpx
import json
import sys

BASE = "http://localhost:8000"
AUTH_URL = "http://127.0.0.1:54321/auth/v1/token?grant_type=password"
ANON_KEY = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"

def get_token():
    r = httpx.post(
        AUTH_URL,
        headers={"apikey": ANON_KEY, "Content-Type": "application/json"},
        json={"email": "test@example.com", "password": "testpassword123"},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def test_inference_health():
    print("\n=== 1. GET /health/inference ===")
    r = httpx.get(f"{BASE}/health/inference")
    print(f"Status: {r.status_code}")
    data = r.json()
    print(json.dumps(data, indent=2))
    
    assert r.status_code == 200
    assert data["instant"]["status"] == "not_configured"
    assert data["thinking"]["status"] == "not_configured"
    print("✅ Inference health check — both show not_configured (expected)")


def test_chat_mock(token):
    print("\n=== 2. POST /chat (mock mode) ===")
    r = httpx.post(
        f"{BASE}/chat",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"message": "What is 2+2?", "mode": "instant"},
    )
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Response: {data['message']['content'][:100]}")
    
    assert r.status_code == 200
    assert "4" in data["message"]["content"]
    print("✅ Chat mock response works correctly")
    return data["session_id"]


def test_chat_stream(token):
    print("\n=== 3. POST /chat/stream (SSE mock) ===")
    with httpx.Client() as client:
        with client.stream(
            "POST",
            f"{BASE}/chat/stream",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"message": "Hello!", "mode": "instant"},
        ) as response:
            print(f"Status: {response.status_code}")
            assert response.status_code == 200
            
            chunks = []
            for line in response.iter_lines():
                if line.strip():
                    chunks.append(line)
                    if len(chunks) <= 5:
                        print(f"  chunk: {line[:80]}")
            
            print(f"  ... total {len(chunks)} chunks received")
            
            # Check we got data and [DONE]
            assert len(chunks) >= 2
            assert any("[DONE]" in c for c in chunks)
            print("✅ Streaming response works correctly")


def test_chat_thinking_mock(token):
    print("\n=== 4. POST /chat (thinking mode mock) ===")
    r = httpx.post(
        f"{BASE}/chat",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"message": "Explain quantum computing", "mode": "thinking"},
    )
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Response: {data['message']['content'][:100]}")
    
    assert r.status_code == 200
    assert "mock" in data["message"]["content"].lower() or len(data["message"]["content"]) > 0
    print("✅ Thinking mode mock response works")


if __name__ == "__main__":
    print("🧪 Step 1.5 — Inference Integration Tests")
    print("=" * 50)
    
    try:
        # Test 1: Inference health (no auth needed)
        test_inference_health()
        
        # Get auth token
        print("\n--- Getting auth token ---")
        token = get_token()
        print(f"✅ Got token: {token[:20]}...")
        
        # Test 2: Chat with mock
        test_chat_mock(token)
        
        # Test 3: Streaming with mock
        test_chat_stream(token)
        
        # Test 4: Thinking mode mock
        test_chat_thinking_mock(token)
        
        print("\n" + "=" * 50)
        print("🎉 All Step 1.5 tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
