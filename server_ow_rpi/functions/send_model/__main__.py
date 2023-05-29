import json
import pickle
import pymongo
import base64

#################
### LIBRARIES ###
#################


###################
#### 0. database.py
from pymongo import MongoClient, ASCENDING
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
    return send_model(params)


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
        # raise 'No Global Model Available'
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

            records = [record for record in
                       training_db.training_clients.find({'globalModelId': global_model_record['id']}).sort('createdAt',
                                                                                                            pymongo.DESCENDING)]
            if len(records) == 0:
                training_clients_id = 1
            else:
                training_clients_id = records[0]['id'] + 1

            training_db.global_models.update_one(
                {'id': global_model_record['id']},
                {'$set': {'currentEpoch': global_model_record['currentEpoch'] + 1, 'serialized': serialized,
                          'updatedAt': datetime.now()}
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
