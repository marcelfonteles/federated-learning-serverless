import json
import pickle
from datetime import datetime
import pymongo

##############################
##############################
from torch import nn
import torch.nn.functional as F


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


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
    return start_training(params)


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
