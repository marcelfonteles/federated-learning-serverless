import pickle
import requests
import base64

if __name__ == '__main__':
    serialized = pickle.dumps({'hello': 'World'})
    b64 = base64.b64encode(serialized).decode('utf-8')
    url = 'https://yz3acob28f.execute-api.us-east-1.amazonaws.com/default/learnPython'
    headers = {"Client-Id": str(42), 'Content-Type': 'application/json'}
    response = requests.post(url, json={'bin': b64}, headers=headers)
    print(base64.b64encode(serialized).decode('utf-8'))
    print(base64.b64encode(serialized))
    print(base64.b64decode(base64.b64encode(serialized).decode('utf-8')))
    print(response.text)
