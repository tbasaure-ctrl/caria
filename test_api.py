"""Test API endpoint"""
import requests
import json

response = requests.post(
    "http://localhost:8000/analyze",
    json={"ticker": "NVDA", "user_query": "está cara?"}
)

print(json.dumps(response.json(), indent=2, ensure_ascii=False))
