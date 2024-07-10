import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "features": [
        1, 500, 2023, 0.5, 5, 0.05, 1000, 800, 600, 300, 50, 10, 8, 6, 3, 1, 20, 0.2, 40000, 1
    ]
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print(f"Next Week's Deaths: {response.json()['prediction']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
