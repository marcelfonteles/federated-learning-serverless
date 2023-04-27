import json
import pickle
from datetime import datetime
import pymongo

from models import CNNMnist, CNNCifar
from database import get_database

def main(params):
    return { 'hello': 'world' }
    # OK (0)
    return start_training(params)


# OK OK
def start_training(event):
    body = json.loads(event['body'])
    dataset = body['dataset']
    number_of_clients = body['numberOfClients']
    frac_to_train = body['fracToTrain']
    if dataset == 'mnist':
        # Training for MNIST
        global_model = CNNMnist()
    else:
        # Training for CIFAR10
        global_model = CNNCifar()
    global_model.train()


    # MongoDB Connection
    training_db = get_database()
    global_models_table = training_db.global_models
    records = [record for record in global_models_table.find().sort('createdAt', pymongo.DESCENDING)]

    if len(records) == 0:  # create a new record on db
        global_weights = global_model.state_dict()
        serialized = pickle.dumps(global_weights)
        global_models_table.insert_one({
            'id': 1,
            'dataset': dataset,
            'serialized': serialized,
            'totalEpochs': 10,
            'currentEpoch': 1,
            'isTraining': True,
            'testAccuracy': 0,
            'numberOfClients': number_of_clients,
            'fracToTrain': frac_to_train,
            'createdAt': datetime.now(),
            'updatedAt': datetime.now(),
        })
    else:  # load or create from db
        record = records[0]  # ordered by createdAt DESCENDING, position 0 is the newest one.
        if record['isTraining']:  # load binary from db
            serialized = record['serialized']
            weights = pickle.loads(serialized)
            global_model.load_state_dict(weights)
        else:  # start a new model
            global_weights = global_model.state_dict()
            serialized = pickle.dumps(global_weights)
            global_models_table.insert_one({
                'id': record['id'] + 1,
                'dataset': dataset,
                'serialized': serialized,
                'totalEpochs': 10,
                'currentEpoch': 1,
                'isTraining': True,
                'testAccuracy': 0,
                'numberOfClients': number_of_clients,
                'fracToTrain': frac_to_train,
                'createdAt': datetime.now(),
                'updatedAt': datetime.now(),
            })

    return {
        "statusCode": 200,
        "body": json.dumps('start_training path')
    }

"""
BUILD STEPS

docker build -t server_ow .
docker tag server_ow marcelfonteles/server_ow
docker push marcelfonteles/server_ow

## Delete action
wsk -i action delete server_ow

## Delete zip file
rm server_ow.zip

## Create zip file
zip -r server_ow.zip __main__.py another
zip -r server_ow.zip ./*

## Deploy action
wsk -i action create server_ow --docker marcelfonteles/server_ow __main__.py
wsk -i action create server_ow --docker marcelfonteles/server_ow __main__.py --web

wsk -i action create server_ow --docker marcelfonteles/server_ow server_ow.zip
wsk -i action create server_ow --docker marcelfonteles/server_ow server_ow.zip --web true

## Test action
wsk -i action invoke server_ow --result
wsk -i action invoke server_ow --result -v

## Crate API route
wsk -i api create /jokes post server_ow --response-type json

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