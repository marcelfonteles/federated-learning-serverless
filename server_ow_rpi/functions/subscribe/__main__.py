import json
import numpy as np
from datetime import datetime
import pymongo

from pymongo import MongoClient, ASCENDING, DESCENDING
import os

##############################
##############################
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
    return subscribe()


# OK OK
def subscribe():
    training_db = get_database()
    # get global model
    records = [record for record in training_db.global_models.find().sort('createdAt', pymongo.DESCENDING)]
    if len(records) == 0:
        print('No Global Model Available (4)')
        return {
            "statusCode": 200,
            "body": json.dumps('No Global Model Available (4)')
        }
        # raise 'No Global Model Available'
    else:
        global_model_record = records[0]
        n_clients = global_model_record['numberOfClients']
        n_clients_to_train = global_model_record['numberOfClients'] * global_model_record['fracToTrain']

    # get clients of this model
    clients = [record for record in
               training_db.clients.find({'globalModelId': global_model_record['id']}).sort('id', pymongo.ASCENDING)]

    if len(clients) < n_clients:
        client_id = len(clients)
        client = {
            'id': client_id,
            'globalModelId': global_model_record['id'],
            'datasetIndexes': [],
            'createdAt': datetime.now(),
            'updatedAt': datetime.now(),
        }

        training_db.clients.insert_one(client)
        clients.append(client)
        if len(clients) == n_clients:

            m = max(int(n_clients_to_train), 1)
            clients_id_to_train = np.random.choice(range(n_clients), m, replace=False).tolist()
            records = [record for record in training_db.training_clients.find().sort('createdAt', pymongo.DESCENDING)]
            if len(records) == 0:
                training_clients_id = 1
            else:
                training_clients_id = records[0]['id'] + 1

            records = [record for record in training_db.global_models.find().sort('createdAt', pymongo.DESCENDING)]
            if len(records) == 0:
                print('No Global Model Available (5)')
                return {
                    "statusCode": 200,
                    "body": json.dumps('No Global Model Available (5)')
                }
                # raise 'No Global Model Available'
            else:
                global_model_id = records[0]['id']

            training_db.training_clients.insert_one({
                'id': training_clients_id,
                'clients': clients_id_to_train,
                'trainedClients': [],
                'globalModelId': global_model_id,
                'currentEpoch': 1,
                'createdAt': datetime.now(),
                'updatedAt': datetime.now(),
            })

        return {
            "statusCode": 200,
            "body": json.dumps({"id": client_id})
        }
    else:
        return {
            "statusCode": 200,
            "body": json.dumps({"id": -1})
        }

    # OK OK

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
