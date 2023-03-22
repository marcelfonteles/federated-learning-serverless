from flask import Flask, request
import pickle
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import copy
import os
import json
import base64
from dotenv import load_dotenv

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from src.models import CNNMnist, CNNCifar
from src.datasets import get_dataset
from src.update import LocalUpdate
from src.utils import logging

load_dotenv(dotenv_path='../.env')
base_url = os.getenv('BASE_URL_OW')

# Subscribing to server
response = requests.post(base_url + 'subscribe', json={"path": "subscribe"}, verify=False)
client_id = json.loads(response.json()['body'])['id']
if client_id == None or client_id == -1:
    raise 'No empty space left.'


# Initialization: get the last version of global model
response = requests.post(base_url + 'get_model', json={"path": "get_model"}, verify=False)
bin_weights = base64.b64decode(json.loads(response.text)['body'])
local_weights = pickle.loads(bin_weights)
n_clients = int(response.json()['headers']['Clients-Number'])

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
get_data_headers = {'Content-Type': 'application/json'}
response = requests.post(base_url + 'get_data', json={"path": "get_data", "client_id": client_id}, headers=get_data_headers, verify=False)
user_group = json.loads(json.loads(response.json()['body'])['dict_user'])
user_group = set(user_group)
train_dataset, test_dataset = get_dataset(dataset, n_clients)


# Log file path
dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, 'logs/' + str(client_id) + '_ow.log')
logging(f'| #### Starting a new client #### |', True, log_path)


# Flask App
app = Flask(__name__)


# Configuring job to verify if they need to train
def train():
    global local_model, local_weights
    headers = {'Content-Type': 'application/json'}
    data = {"client_id": client_id}

    # Check if it is time to train
    res = requests.post(base_url + 'get_clients_to_train', json=data, headers=headers, verify=False)
    res_json = json.loads(res.json()['body'])
    must_train = res_json['train']
    logging(f'| client_id: {client_id}, must_train: {must_train} |', True, log_path)
    if must_train:
        global_epoch = res_json['epoch']

        # Get the newest global model
        res = requests.post(base_url + 'get_model', verify=False)
        bin_local_weights = base64.b64decode(json.loads(res.text)['body'])
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
        res = requests.post(url, json={'model': b64}, headers=headers, verify=False)
        print(res)

train()
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
    # TODO: implement
    return {"hello": "world", "array": [1, 2, 3]}


# This route can be used to test to model prediction in client
# Todo: Add a simple interface so the user can test.
@app.route("/predict", methods=['POST'])
def predict():
    image = request.files['image']
    img = Image.open(image)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    pred = torch.argmax(local_model(img_tensor))
    return {"prediction": pred.item()}
