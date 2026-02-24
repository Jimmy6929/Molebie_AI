#!/bin/bash
# Test script for gateway chat endpoint

# Get a fresh token
TOKEN=$(curl -s -X POST 'http://127.0.0.1:54321/auth/v1/token?grant_type=password' \
  -H "apikey: sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "testpassword123"}' | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

echo "Token: ${TOKEN:0:50}..."
echo ""

# Test chat endpoint
echo "Testing POST /chat..."
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! What is 2+2?"}'

echo ""
echo ""
echo "Testing GET /chat/sessions..."
curl -s http://127.0.0.1:8000/chat/sessions \
  -H "Authorization: Bearer $TOKEN"

echo ""
