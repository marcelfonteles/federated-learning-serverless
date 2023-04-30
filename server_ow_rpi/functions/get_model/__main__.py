import json
import pymongo
import base64

##############################
##############################
from pymongo import MongoClient, ASCENDING, DESCENDING
import os

def get_database():
    try:
        client = MongoClient(os.environ['MONGODB_URL'])
    except Exception as e:
        client = MongoClient('localhost', 27017, username='root', password='password')
    return client.training


def get_last_global_model():
    client = get_database()
    training = client.training
    global_models = training.global_models
    records = [record for record in global_models.find().sort('createdAt', ASCENDING)]
    client.close()


##############################
##############################


#################
### MAIN CODE ###
#################
def main(params):
    return get_model()


def get_model():
    training_db = get_database()
    records = [record for record in training_db.global_models.find().sort('createdAt', pymongo.DESCENDING)]
    if len(records) == 0:
        print('No Global Model Available (3)')
        return {
            "statusCode": 200,
            "body": json.dumps('No Global Model Available (3)')
        }
        # raise 'No Global Model Available'
    else:
        serialized = records[0]['serialized']

    return {
        'headers': {"Content-Type": "application/octet-stream", 'Clients-Number': records[0]['numberOfClients']},
        'statusCode': 200,
        'body': base64.b64encode(serialized).decode('utf-8')
    }


"""
BUILD STEPS

## Build Dockerfile. Will be used as RUNTIME
docker build -t server_ow .
docker tag server_ow marcelfonteles/server_ow
docker push marcelfonteles/server_ow

## Delete action
### If there is an action created with the same same, you need to delete it
wsk -i action delete server_ow

## Delete zip file
### If there is an zip file with the same name
rm server_ow.zip

## Create zip file
zip -r server_ow.zip ./*

## Deploy action
### Deploy an action with only one file
wsk -i action create server_ow --docker marcelfonteles/server_ow __main__.py --web

### Deploy an action with multiple files
wsk -i action create server_ow --docker marcelfonteles/server_ow server_ow.zip --web true

## Test action
wsk -i action invoke server_ow --result
wsk -i action invoke server_ow --result -v

## All steps together
wsk -i action delete server_ow && rm server_ow.zip && zip -r server_ow.zip ./* && wsk -i action create server_ow --docker marcelfonteles/server_ow server_ow.zip --web true
wsk -i api create /root post server_ow --response-type json

wsk -i api create /send_model post server_ow --response-type json
wsk -i api create /get_clients_to_train post server_ow --response-type json
wsk -i api create /get_data post server_ow --response-type json
wsk -i api create /get_model post server_ow --response-type json
wsk -i api create /subscribe post server_ow --response-type json
wsk -i api create /start_training post server_ow --response-type json
"""
