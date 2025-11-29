import requests
import sys

API_KEY = "3KHQX7KNMNT7H7MZ"

def test_alpha_vantage():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=technology&apikey={API_KEY}"
    try:
        print(f"Testing Alpha Vantage API with key ending in ...{API_KEY[-4:]}")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "feed" in data:
                print("SUCCESS: Received news feed.")
                print(f"Items: {len(data['feed'])}")
                print("Sample Title:", data['feed'][0]['title'])
            elif "Note" in data:
                print("WARNING: API Limit reached or other note:", data['Note'])
            elif "Error Message" in data:
                print("ERROR:", data['Error Message'])
            else:
                print("UNKNOWN RESPONSE:", data.keys())
        else:
            print(f"HTTP ERROR: {response.status_code}")
    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    test_alpha_vantage()
