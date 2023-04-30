import json
from datetime import datetime
import pymongo


##############################
##############################
from torchvision import datasets, transforms
import numpy as np


def get_test_dataset(dataset):
    if dataset == 'mnist':
        data_dir = '/tmp/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    elif dataset == 'cifar10':
        data_dir = '/tmp/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

    return test_dataset


def get_user_group(dataset, num_users, dict_users):
    if dataset == 'mnist':
        data_dir = '/tmp/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
    elif dataset == 'cifar10':
        data_dir = '/tmp/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

    dict_user = iid_sampling(train_dataset, num_users, dict_users)

    return dict_user


def iid_sampling(dataset, num_users, dict_users):
    num_items = int(len(dataset)/num_users)
    dict_user, all_idxs = [], [i for i in range(len(dataset))]

    # Remove all selected data from randomly choose
    for i in range(num_users):
        if i in dict_users:
            all_idxs = list(set(all_idxs) - dict_users[i])

    dict_user = set(np.random.choice(all_idxs, num_items, replace=False))

    return dict_user


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
    return get_data(params)


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
        # raise 'No Global Model Available'
    else:
        global_model_record = records[0]

    # get all clients
    clients = [record for record in
               training_db.clients.find({'globalModelId': global_model_record['id']}).sort('id', pymongo.ASCENDING)]
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
