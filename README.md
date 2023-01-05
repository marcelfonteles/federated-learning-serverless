# Federated Learning Serverless
This project will use the AWS Lambda to run the `server/app.py`

### Requirements
Install all the packages from requirments.txt

- flask
- torch
- torchvision
- numpy
- tensorboardx
- matplotlib
- apscheduler
- Pillow
```
pip3 install -r client/requirements.txt
pip3 install -r server/requirements.txt
```

### Models and Dataset
Currently, this project uses CNN with 2 layers and two types of dataset: MNIST and CIFAR10. We use `SGD` as optimizer

### How does this project works
We have two mainly pieces of code, one represents the server and other represents the client (the entity who owns 
the data). So firstly we must start one server and after that we initialize multiple clients (this will 
represent multiple users).

After that, all client subscribe to the server and get a subset of the dataset, this is one way to guarantee that all
users have different data. The server do not see the data and do not train any model with the data. The server 
only randomly choose a subset of dataset and send this information to the user.

For default, we will use 20 clients and the dataset will be divided equally to all the users.

After the division of the dataset, the server will randomly choose 10 clients at each epoch to train their local model.
On the client, the local model will be trained using CNN and 10 epochs, after the training each local model will be
sent to the server and then the fedAvg algorithm will be executed to generate a new global model. 
This process will repeat 10 times and after that we will have a global model ready to use for predictions for new data.

### Building the experiments
If you are on a Mac M1, you must specify the platform
```
docker buildx build --platform linux/amd64 -t fourth-serverless .
docker build --platform linux/amd64 -t fifth-serverless .
```

```
aws ecr create-repository --repository-name fifth-serverless
	
aws_region=us-east-1
aws_account_id=


aws ecr get-login-password \
--region $aws_region \
| docker login \
--username AWS \
--password-stdin $aws_account_id.dkr.ecr.$aws_region.amazonaws.com

docker tag fifth-serverless $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/fifth-serverless

docker push $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/fifth-serverless
```

Get image URI and add to the `serverless.yml` file and then execute the following command:
```
serverless deploy
```



### Running the experiments
