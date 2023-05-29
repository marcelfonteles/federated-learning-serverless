import json
import pymongo

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
    return get_clients_to_train(params)


def get_clients_to_train(event):
    training_db = get_database()
    client_id = event['client_id']

    records = [record for record in training_db.global_models.find().sort('createdAt', pymongo.DESCENDING)]
    if len(records) == 0:
        print('No Global Model Available (1)')
        return {
            "statusCode": 200,
            "body": json.dumps('No Global Model Available (1)')
        }
        # raise 'No Global Model Available'
    else:
        global_model_record = records[0]
        global_current_epoch = records[0]['currentEpoch']

    records = [record for record in
               training_db.training_clients.find({'globalModelId': global_model_record['id']}).sort('id',
                                                                                                    pymongo.DESCENDING)]
    if len(records) == 0 or global_model_record['isTraining'] == False:
        return {
            "statusCode": 200,
            "body": json.dumps({"train": False})
        }
    else:
        training_clients_id = records[0]['id']
        clients_to_train = records[0]['clients']
        trained_clients = records[0]['trainedClients']
        clients_to_train_epoch = records[0]['currentEpoch']

    if clients_to_train.count(client_id) and global_current_epoch == clients_to_train_epoch:
        clients_to_train.remove(client_id)
        trained_clients.append(client_id)
        training_db.training_clients.update_one(
            {'id': training_clients_id},
            {'$set': {'clients': clients_to_train, 'trainedClients': trained_clients}}
        )
        return {
            "statusCode": 200,
            "body": json.dumps({"train": True, "epoch": global_current_epoch})
        }
    else:
        return {
            "statusCode": 200,
            "body": json.dumps({"train": False})
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
