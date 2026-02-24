#!/usr/bin/env python3
"""Test script for gateway chat endpoints."""

import json
import httpx

def main():
    # Get token
    with open('/tmp/auth_response.json') as f:
        token = json.load(f)['access_token']

    headers = {'Authorization': f'Bearer {token}'}
    base_url = 'http://127.0.0.1:8000'

    print("=== Testing GET /chat/sessions ===")
    response = httpx.get(f'{base_url}/chat/sessions', headers=headers)
    print(f"Status: {response.status_code}")
    sessions = response.json()
    print(f"Sessions: {json.dumps(sessions, indent=2)}")

    if sessions.get('sessions'):
        session_id = sessions['sessions'][0]['id']
        
        print(f"\n=== Testing GET /chat/sessions/{session_id}/messages ===")
        response = httpx.get(f'{base_url}/chat/sessions/{session_id}/messages', headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Messages: {json.dumps(response.json(), indent=2)}")
        
        print(f"\n=== Testing another message in same session ===")
        response = httpx.post(
            f'{base_url}/chat',
            headers=headers,
            json={'message': 'Can you explain that?', 'session_id': session_id}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == '__main__':
    main()
