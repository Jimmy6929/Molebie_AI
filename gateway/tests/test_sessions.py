"""
Test script for Step 1.7 — Session Management.
Run: python test_sessions.py
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


def headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def test_create_session(token):
    print("\n=== 1. Create new session (via POST /chat) ===")
    r = httpx.post(f"{BASE}/chat", headers=headers(token),
                   json={"message": "Session management test", "mode": "instant"})
    assert r.status_code == 200
    data = r.json()
    sid = data["session_id"]
    print(f"  Created session: {sid}")
    print(f"  Title: {data.get('session_title')}")
    print("✅ Session created")
    return sid


def test_list_sessions(token):
    print("\n=== 2. List sessions ===")
    r = httpx.get(f"{BASE}/chat/sessions", headers=headers(token))
    assert r.status_code == 200
    data = r.json()
    count = len(data["sessions"])
    print(f"  Found {count} session(s)")
    for s in data["sessions"][:3]:
        print(f"    - {s['title']} ({s['id'][:8]}...)")
    print("✅ Sessions listed")
    return data["sessions"]


def test_switch_session(token, session_id):
    print(f"\n=== 3. Switch to session {session_id[:8]}... ===")
    r = httpx.get(f"{BASE}/chat/sessions/{session_id}/messages", headers=headers(token))
    assert r.status_code == 200
    msgs = r.json()
    print(f"  Found {len(msgs)} message(s)")
    print("✅ Session switched (messages loaded)")


def test_rename_session(token, session_id):
    print(f"\n=== 4. Rename session {session_id[:8]}... ===")
    r = httpx.patch(f"{BASE}/chat/sessions/{session_id}",
                    headers=headers(token),
                    json={"title": "Renamed: My Test Session"})
    print(f"  Status: {r.status_code}")
    assert r.status_code == 200
    data = r.json()
    print(f"  New title: {data['title']}")
    assert data["title"] == "Renamed: My Test Session"
    print("✅ Session renamed")


def test_send_to_existing(token, session_id):
    print(f"\n=== 5. Send message to existing session ===")
    r = httpx.post(f"{BASE}/chat", headers=headers(token),
                   json={"message": "Second message in session", "mode": "instant",
                          "session_id": session_id})
    assert r.status_code == 200
    print(f"  Response: {r.json()['message']['content'][:60]}")
    
    # Check messages count
    r2 = httpx.get(f"{BASE}/chat/sessions/{session_id}/messages", headers=headers(token))
    msgs = r2.json()
    print(f"  Total messages now: {len(msgs)}")
    assert len(msgs) >= 4  # 2 user + 2 assistant
    print("✅ Messages persist across interactions")


def test_multiple_sessions(token):
    print(f"\n=== 6. Multiple independent sessions ===")
    # Create second session
    r = httpx.post(f"{BASE}/chat", headers=headers(token),
                   json={"message": "This is session two", "mode": "instant"})
    sid2 = r.json()["session_id"]
    print(f"  Created session 2: {sid2[:8]}...")
    
    # Verify sessions are independent
    r2 = httpx.get(f"{BASE}/chat/sessions/{sid2}/messages", headers=headers(token))
    msgs = r2.json()
    assert len(msgs) == 2  # 1 user + 1 assistant
    print(f"  Session 2 has {len(msgs)} messages (independent)")
    print("✅ Multiple sessions work independently")
    return sid2


def test_delete_session(token, session_id):
    print(f"\n=== 7. Delete session {session_id[:8]}... ===")
    r = httpx.delete(f"{BASE}/chat/sessions/{session_id}", headers=headers(token))
    assert r.status_code == 204
    
    # Verify deleted
    r2 = httpx.get(f"{BASE}/chat/sessions/{session_id}/messages", headers=headers(token))
    assert r2.status_code == 404
    print("✅ Session deleted and verified gone")


if __name__ == "__main__":
    print("🧪 Step 1.7 — Session Management Tests")
    print("=" * 50)

    try:
        token = get_token()
        print(f"✅ Authenticated: {token[:20]}...")

        # 1. Create
        sid1 = test_create_session(token)

        # 2. List
        test_list_sessions(token)

        # 3. Switch
        test_switch_session(token, sid1)

        # 4. Rename
        test_rename_session(token, sid1)

        # 5. Send to existing
        test_send_to_existing(token, sid1)

        # 6. Multiple sessions
        sid2 = test_multiple_sessions(token)

        # 7. Delete
        test_delete_session(token, sid2)

        # Final list
        print("\n=== Final session list ===")
        sessions = test_list_sessions(token)

        print("\n" + "=" * 50)
        print("🎉 All Step 1.7 tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
