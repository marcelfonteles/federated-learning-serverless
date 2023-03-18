from flask import Flask, request
import pickle
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import copy
import os
import json
import base64

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from src.models import CNNMnist, CNNCifar
from src.datasets import get_dataset
from src.update import LocalUpdate
from src.utils import logging

base_url = 'https://plxyyoan0c.execute-api.us-east-1.amazonaws.com/dev/'
# Subscribing to server
client_id = -1
response = requests.post(base_url + 'subscribe', json={"path": "subscribe"})
client_id = response.json()['id']
if client_id == None:
    raise 'No empty space left.'


# Initialization: get the last version of global model
response = requests.post(base_url + 'get_model', json={"path": "get_model"})
bin_weights = base64.b64decode(response.text)
local_weights = pickle.loads(bin_weights)
n_clients = int(response.headers['Clients-Number'])

dataset = 'mnist'  # or mnist
if dataset == 'mnist':
    # Training for MNIST
    local_model = CNNMnist()
else:
    # Training for CIFAR10
    local_model = CNNCifar()

local_model.load_state_dict(local_weights)
local_model.train()


# Initialization: randomly select the data from dataset for this client
headers = {'Content-Type': 'application/json'}
response = requests.post(base_url + 'get_data', json={"path": "get_data", "client_id": client_id}, headers=headers)
user_group = response.json()['dict_user']
user_group = set(json.loads(user_group))
train_dataset, test_dataset = get_dataset(dataset, n_clients)


# Log file path
dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, 'logs/' + str(client_id) + '.log')
logging(f'| #### Starting a new client #### |', True, log_path)


# Flask App
app = Flask(__name__)


# Configuring job to verify if they need to train
def train():
    global local_model, local_weights
    headers = {'Content-Type': 'application/json'}
    data = {"client_id": client_id}
    res = requests.post(base_url + 'get_clients_to_train', json=data, headers=headers)
    res_json = res.json()
    must_train = res_json['train']
    logging(f'| client_id: {client_id}, must_train: {must_train} |', True, log_path)
    if must_train:
        global_epoch = res_json['epoch']
        # Get the newest global model
        res = requests.post(base_url + 'get_model')
        bin_local_weights = base64.b64decode(res.content)
        local_weights = pickle.loads(bin_local_weights)
        if dataset == 'mnist':
            local_model = CNNMnist()
        elif dataset == 'cifar10':
            local_model = CNNCifar()
        local_model.load_state_dict(local_weights)
        local_model.train()

        # Get the training parameters (batch_size, n_epochs)
        local = LocalUpdate(dataset=train_dataset, idxs=user_group)
        w, loss = local.update_weights(model=copy.deepcopy(local_model), global_round=global_epoch, client=client_id, log_path=log_path)

        # New local model with updated weights
        local_model.load_state_dict(w)

        # Send new local model to server
        serialized = pickle.dumps(w)
        b64 = base64.b64encode(serialized).decode('utf-8')
        url = base_url + 'send_model'
        headers = {"client-id": str(client_id), 'Content-Type': 'application/json'}
        res = requests.post(url, json={'model': b64}, headers=headers)
        print(res)


scheduler = BackgroundScheduler()
scheduler.add_job(func=train, trigger="interval", seconds=15)
scheduler.start()


# Healthcheck route
@app.route("/", methods=['GET', 'POST'])
def home():
    return {"message": "Client is running."}


# This route is used to trigger the client to train their local model.
# When called this endpoint we must send n_epochs and batch_size.
# The response is the update weights for local model.
@app.route("/train", methods=['POST'])
def train():
    return {"hello": "world", "array": [1, 2, 3], "nested": {"again": 1}}


# This route can be used to test to model prediction in client
# Todo: Add a simple interface so the user can test.
@app.route("/predict", methods=['POST'])
def predict():
    image = request.files['image']
    img = Image.open(image)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    pred = torch.argmax(local_model(img_tensor))
    return {"prediction": pred.item()}
