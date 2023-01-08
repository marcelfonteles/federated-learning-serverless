import pickle
import requests
import base64

if __name__ == '__main__':

    ## Get Model
    # url = 'https://plxyyoan0c.execute-api.us-east-1.amazonaws.com/dev/get_model'
    # response = requests.post(url)
    # b64 = base64.b64decode(response.text)
    # weights = pickle.loads(b64)
    # print(weights)

    ## Get Data
    body = {'client_id': 0}
    headers = {'Content-Type': 'application/json'}
    url = 'https://plxyyoan0c.execute-api.us-east-1.amazonaws.com/dev/get_data'
    response = requests.post(url, json=body, headers=headers)
    print(response.text)
    
    
    # serialized = pickle.dumps({'hello': 'World'})
    # b64 = base64.b64encode(serialized).decode('utf-8')
    # url = 'https://plxyyoan0c.execute-api.us-east-1.amazonaws.com/dev/get_model'
    # headers = {"Client-Id": str(42), 'Content-Type': 'application/json'}
    # response = requests.post(url, json={'bin': b64}, headers=headers)
    # print(base64.b64encode(serialized).decode('utf-8'))
    # print(base64.b64encode(serialized))
    # print(base64.b64decode(base64.b64encode(serialized).decode('utf-8')))
    # print(response.text)
