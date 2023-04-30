import json
import pickle
from datetime import datetime
import pymongo

# I need to put all code in one file, because if i put all in separated files
# the raspberry pi can not run.

#################
### LIBRARIES ###
#################


###################
#### 0. database.py
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


###################
#### 1. datasets.py
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


##################
##### 2. models.py
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


#################
##### 3. utils.py
import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
from datetime import datetime


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def test_inference(model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


def logging(message, write_to_file, filepath):
    print(message)
    if write_to_file:
        try:
            f = open(filepath, 'a')
            f.write('[%s] %s \n' % (datetime.now(), message))
            f.close()
        except Exception as e:
            print(e)


########################
### END OF LIBRARIES ###
########################


#################
### MAIN CODE ###
#################
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

OBS: to make it work the apigateway, you have to build the docker image using the RPi: https://github.com/apache/openwhisk-apigateway
     Build using `make build`


alsto
"""