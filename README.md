# Federated Learning Serverless
Esse projeto implementa uma arquitetura Serverless para o uso em Aprendizado Federado

### Requisitos
Instale os pacotes necessários:

- Para o servidor
```
torch
torchvision
numpy
pymongo
```

- Para o cliente
```
APScheduler
Pillow
Flask
Pickle
requests
numpy
torch
```

```
pip install -r clients/requirements.txt
pip install -r server_ow/requirements.txt
pip install -r server_aws/requirements.txt
```

### Modelos e dataset
Atualmente esse projeto tem compatibilidade para o uso de CNN de duas camadas e dois dataset: MNIST e CIFAR10. É utilizado
o `SGD` como otimizador.

### Como esse projeto funciona
Temos duas peças principais de código, uma representa o servidor e a outra representa o cliente (a entidade que detém 
os dados). Assim, em primeiro lugar, temos de iniciar um servidor e, depois disso, inicializamos vários clientes (isto 
representam múltiplos utilizadores).

Depois disso, todos os clientes subscrevem no servidor e recebem um subconjunto do conjunto de dados, 
esta é uma forma de garantir que todos os utilizadores têm dados diferentes. 
O servidor não vê os dados e não treina nenhum modelo com os dados. O servidor 
apenas escolhe aleatoriamente um subconjunto de dados e envia esta informação ao utilizador.

Por padrão, utilizaremos 20 clientes e o conjunto de dados será dividido igualmente para todos os utilizadores.

Após a divisão do conjunto de dados, o servidor irá escolher aleatoriamente 10 clientes em cada epoch para treinar o seu modelo local.
No cliente, o modelo local será treinado utilizando a CNN e 10 epochs, após o treinamento, cada modelo local será
enviado para o servidor e depois o algoritmo `fedAvg` será executado para gerar um novo modelo global. 
Este processo irá repetir-se 10 vezes e depois disso teremos um modelo global pronto a usar para previsões de novos dados.

### Executando os experimentos (AWS Lambda)
#### Atenção: se você estiver utilizando um processador ARM (por exemplo, Mac M1), você precisa especificar a plataforma
```
docker buildx build --platform linux/amd64 -t image-name .
docker build --platform linux/amd64 -t image-name .
```
---
Docker build
```
docker build -t server-serverless .
```

Local test
```
docker run --rm -p 8080:8080 server-serverless
```


Create ECR Repository
```
aws ecr create-repository --repository-name server-serverless
```

Define shell environment variables
```
aws_region=us-east-1
aws_account_id=

aws_region=us-east-1 && aws_account_id=
```

Login
```
aws ecr get-login-password \
--region $aws_region \
| docker login \
--username AWS \
--password-stdin $aws_account_id.dkr.ecr.$aws_region.amazonaws.com
```

Create tag
```
docker tag server-serverless $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/server-serverless
```
Docker Push
```
docker push $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/server-serverless
```

Get image URI and add to the `serverless.yml` file and then execute the following command:
```
serverless deploy
```

```
docker tag server-serverless $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/server-serverless && \
docker push $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/server-serverless && \
serverless deploy
```

### Running the experiments (AWS Lambda)
You must have access to one mongodb database and fill the `MONGODB_URL` in Dockerfile

Firstly, you must create one AWS Lambda function with the Dockerfile in this repository, and you could do that by
following this article: https://www.serverless.com/blog/deploying-pytorch-model-as-a-serverless-service

After deployed the lambda function you can run the following command
```
sh client/start.sh
```

### Running the experiments (Apache OpenWhisk)
To do: trying to make it work 

### Results
MNIST Test Accuracy using federated learning strategy: 96.67%
