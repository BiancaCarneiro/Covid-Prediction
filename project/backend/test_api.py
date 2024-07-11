import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "features": [
        -0.00780999,  0.41073615,  0.01019074,  0.13420779,  0.664905  ,
        -0.13138404, -0.12870815, -0.12098973, -0.09076258,  0.36131824,
        -0.45460783, -0.47667697, -0.44810676, -0.26483611,  3.66606401,
         0.86114141,  5.80604221,  0.56813726, -1.69545159, -0.40005454,
        -0.30660269, -2.3141746 , -0.07167211, -0.07244861, -0.07343516,
        -0.07294352,  0.88850864, -0.62147068
    ]
}

expected_result = 1490.0

response = requests.post(url, json=data)

if response.status_code == 200:
    print(f"Next Week's Deaths: {response.json()['prediction']}")
    print(f"Expected Next Week's Deaths: {expected_result}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
