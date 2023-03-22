import json
import pickle
import numpy as np
from datetime import datetime
import pymongo
import base64

from src.models import CNNMnist, CNNCifar
from src.utils import average_weights, test_inference
from src.datasets import get_test_dataset, get_user_group
from src.database import get_database

def main(params):
    route = params['__ow_headers']['x-forwarded-url']
    if 'send_model' in route:
        # OK (5)
        return send_model(params)
    elif 'get_clients_to_train' in route:
        # OK (4)
        return get_clients_to_train(params)
    elif 'get_data' in route:
        # OK (3)
        return get_data(params)
    elif 'get_model' in route:
        # OK (2)
        return get_model()
    elif 'subscribe' in route:
        # OK (1)
        return subscribe()
    elif 'start_training' in route:
        # OK (0)
        return start_training(params)
    else:
        # OK
        return home()


# OK
def send_model(event):
    training_db = get_database()
    # Insert model data in local model table
    headers = event['__ow_headers']
    client_id = int(headers['client-id'])

    model = event['model']
    serialized = base64.b64decode(model)
    content = pickle.loads(serialized)

    records = [record for record in training_db.local_models.find().sort('id', pymongo.DESCENDING)]
    if len(records) == 0:
        local_model_id = 1
    else:
        local_model_id = records[0]['id'] + 1

    records = [record for record in training_db.global_models.find().sort('id', pymongo.DESCENDING)]
    if len(records) == 0:
        print('No Global Model Available (0)')
        return {
                "statusCode": 200,
                "body": json.dumps('No Global Model Available (0)')
        }    
    else:
        global_model_record = records[0]
        n_clients = global_model_record['numberOfClients']
        n_clients_to_train = global_model_record['numberOfClients'] * global_model_record['fracToTrain']

    find_query = {
        'clientId': client_id,
        'globalModelId': global_model_record['id'],
    }
    records = [record for record in training_db.local_models.find(find_query).sort('id', pymongo.DESCENDING)]
    if len(records) == 0:
        training_db.local_models.insert_one({
            'id': local_model_id,
            'clientId': client_id,
            'globalModelId': global_model_record['id'],
            'model': serialized,
            'epoch': global_model_record['currentEpoch'],
            'createdAt': datetime.now(),
            'updatedAt': datetime.now(),
        })
    else:
        training_db.local_models.update_one(
            {'clientId': client_id, 'globalModelId': global_model_record['id']},
            {'$set': {'model': serialized, 'epoch': global_model_record['currentEpoch'], 'updatedAt': datetime.now()}}
        )

    # This code bellow must go to another endpoint or go to a cronjob, so we can guarantee that will create only one record for epoch.
    clients_weights = []
    find_query = {
        'epoch': global_model_record['currentEpoch'],
        'globalModelId': global_model_record['id'],
    }
    records = [record for record in training_db.local_models.find(find_query).sort('id', pymongo.DESCENDING)]
    for record in records:
        weights = pickle.loads(record['model'])
        clients_weights.append(weights)

    if len(clients_weights) == n_clients_to_train:
        # fedAvg
        # update global weights
        current_epoch = global_model_record['currentEpoch']
        n_epochs = global_model_record['totalEpochs']
        print('Fazendo média | epoch {}'.format(current_epoch))
        global_weights = average_weights(clients_weights)
        serialized = pickle.dumps(global_weights)

        # update global weights
        if global_model_record['dataset'] == 'mnist':
            # Training for MNIST
            global_model = CNNMnist()
        else:
            # Training for CIFAR10
            global_model = CNNCifar()
        global_model.train()
        global_model.load_state_dict(global_weights)

        # random select new clients
        if current_epoch < n_epochs:
            print('Escolhendo novos clientes | epoch {}'.format(current_epoch))
            m = max(int(n_clients_to_train), 1)
            clients_id_to_train = np.random.choice(range(n_clients), m, replace=False).tolist()

            records = [record for record in training_db.training_clients.find({'globalModelId': global_model_record['id']}).sort('createdAt', pymongo.DESCENDING)]
            if len(records) == 0:
                training_clients_id = 1
            else:
                training_clients_id = records[0]['id'] + 1

            training_db.global_models.update_one(
                {'id': global_model_record['id']},
                {'$set': {'currentEpoch': global_model_record['currentEpoch'] + 1, 'serialized': serialized, 'updatedAt': datetime.now()}
            })

            training_db.training_clients.insert_one({
                'id': training_clients_id,
                'clients': clients_id_to_train,
                'trainedClients': [],
                'globalModelId': global_model_record['id'],
                'currentEpoch': global_model_record['currentEpoch'] + 1,
                'createdAt': datetime.now(),
                'updatedAt': datetime.now(),
            })
        else:
            training_db.global_models.update_one(
                {'id': global_model_record['id']},
                {'$set': {'isTraining': False, 'updatedAt': datetime.now()}
            })
            print('training is complete')
            test_dataset = get_test_dataset(global_model_record['dataset'])
            test_acc, test_loss = test_inference(global_model, test_dataset)

            print(' \n Results after {} global rounds of training:'.format(current_epoch))
            # print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
            training_db.global_models.update_one(
                {'id': global_model_record['id']},
                {'$set': {'testAccuracy': test_acc, 'updatedAt': datetime.now()}
            })

    return {
                "statusCode": 200,
                "body": json.dumps('send_model finish')
            }    


# OK
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
    else:
        global_model_record = records[0]
        global_current_epoch = records[0]['currentEpoch']

    records = [record for record in training_db.training_clients.find({'globalModelId': global_model_record['id']}).sort('id', pymongo.DESCENDING)]
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


# OK OK


# OK
def get_data(event):
    training_db = get_database()
    client_id = event['client_id']
    # get global model
    records = [record for record in training_db.global_models.find().sort('createdAt', pymongo.DESCENDING)]
    if len(records) == 0:
        print('No Global Model Available (2)')
        return {
                "statusCode": 200,
                "body": json.dumps('No Global Model Available (2)')
            } 
    else:
        global_model_record = records[0]
    
    # get all clients
    clients = [record for record in training_db.clients.find({'globalModelId': global_model_record['id']}).sort('id', pymongo.ASCENDING)]
    client = training_db.clients.find_one({'globalModelId': global_model_record['id'], 'id': client_id})
    dict_users = {}
    for c in clients:
        if c['datasetIndexes'] != []:
            dict_users[c['id']] = set(json.loads(c['datasetIndexes']))

    dict_user = get_user_group(global_model_record['dataset'], global_model_record['numberOfClients'], dict_users)
    training_db.clients.update_one(
        {'globalModelId': global_model_record['id'], 'id': client['id']},
        {'$set': {'datasetIndexes': str(list(dict_user)), 'updatedAt': datetime.now()}
    })
    return {
                "statusCode": 200,
                "body": json.dumps({"dict_user": str(list(dict_user))})
        } 


# OK
def get_model():
    training_db = get_database()
    records = [record for record in training_db.global_models.find().sort('createdAt', pymongo.DESCENDING)]
    if len(records) == 0:
        print('No Global Model Available (3)')
        return {
                "statusCode": 200,
                "body": json.dumps('No Global Model Available (3)')
            }  
    else:
        serialized = records[0]['serialized']

    return {
            'headers': { "Content-Type": "application/octet-stream", 'Clients-Number': records[0]['numberOfClients'] },
            'statusCode': 200,
            'body': base64.b64encode(serialized).decode('utf-8')
    }


# OK
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
    else:
        global_model_record = records[0]
        n_clients = global_model_record['numberOfClients']
        n_clients_to_train = global_model_record['numberOfClients'] * global_model_record['fracToTrain']

    # get clients of this model
    clients = [record for record in training_db.clients.find({'globalModelId': global_model_record['id']}).sort('id', pymongo.ASCENDING)]


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


# OK
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


# OK
def home():
    return {
        "statusCode": 200,
        "body": json.dumps('Home path')
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