import json

def server(event, context):
    if 'get_data' in event['path']:
        return get_data()
    elif 'get_clients_to_train' in event['path']:
        return get_clients_to_train()
    elif 'get_model' in event['path']:
        return get_model()
    elif 'send_model' in event['path']:
        return send_model()
    elif 'subscribe' in event['path']:
        return subscribe()
    elif 'start_training' in event['path']:
        return start_training()
    else:
        return home()
    

    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """

def home():
    return {
        "statusCode": 200,
        "body": json.dumps('Home path')
    }

def get_data():
    return {
        "statusCode": 200,
        "body": json.dumps('Home path')
    }

def get_clients_to_train():
    return {
        "statusCode": 200,
        "body": json.dumps('get_clients_to_train path')
    }

def get_model():
    return {
        "statusCode": 200,
        "body": json.dumps('get_model path')
    }

def send_model():
    return {
        "statusCode": 200,
        "body": json.dumps('send_model path')
    }

def subscribe():
    return {
        "statusCode": 200,
        "body": json.dumps('subscribe path')
    }

def start_training():
    return {
        "statusCode": 200,
        "body": json.dumps('start_training path')
    }