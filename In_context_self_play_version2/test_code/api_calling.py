import json

import requests


def test_call_agent():
    url = "http://60.251.182.99:5487/customized_model"
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "你好"
            },
            {
                "role": "user",
                "content": "我想要訂位"
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # 印出回應狀態與回應內容
    print("Response status code:", response.status_code)
    try:
        response_data = response.json()
        print("Response JSON:", json.dumps(response_data, indent=4, ensure_ascii=False))
    except Exception as e:
        print("Error parsing response JSON:", e)
        print("Response text:", response.text)

if __name__ == '__main__':
    test_call_agent()
