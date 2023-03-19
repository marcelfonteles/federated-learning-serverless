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
python-dotenv
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
##### Atenção: se você estiver utilizando um processador ARM (por exemplo, Mac M1), você precisa especificar a plataforma
```
docker buildx build --platform linux/amd64 -t image-name .
docker build --platform linux/amd64 -t image-name .
```
---
0. Variáveis de ambiente
    Você deve criar um arquivo `.env` a partir de `.env.template` e preencher com suas variáveis de ambiente.
        - `BASE_URL_AWS`: ao executar os experimentos com a AWS, será essa rota que o `cliente aws` irá fazer todos os requisições para o servidor. Irei mostrar mais abaixo como iremos obter o valor dessa variável
        - `BASE_URL_OW`: ao executar os experimentos com o Apache OpenWhisk, será essa rota que o `cliente aws` irá fazer todos os requisições para o servidor. Irei mostrar mais abaixo como iremos obter o valor dessa variável
        - `MONGODB_URL`: rota para o banco da dados que serão salvas as informações do experimento. Você já pode criar uma instância local ou utilizar o plano gratuito do [mongodb](https://www.mongodb.com/).
    
    ```bash
    # Clients
    BASE_URL_AWS=
    BASE_URL_OW=

    # Servers
    MONGODB_URL=
    ```

    Depois de definir as variáveis de ambiente nesse arquivo, execute o comando abaixo na raiz do projeto
    ```bash
    source .env
    ```

   
1. Docker build
    Faça o build da imagem docker do servidor para AWS. Execute o comando abaixo dentro do diretório `server_aws`

    ```bash
    docker build -t server-serverless .
    ```
    
    Ou execute o comando abaixo na raiz do projeto
    ```bash
    docker build -t server-serverless server_aws
    ```

    Caso deseje executar testes na sua máquina, execute o comando abaixo 
    ```bash
    docker run --rm -p 8080:8080 server-serverless
    ```

    E a rota abaixo estará diponível para uso, assim você pode definir essa rota como variável de ambiente `BASE_URL_AWS` e poderá executar todos os testes locais sem a necessidade de fazer o deploy de uma função serverless na AWS:
    ```
    http://localhost:8080/2015-03-31/functions/function/invocations
    ```


2. Create ECR Repository
   É necessário criar um repositório no serviço ECR da AWS. Iremos subir a imagem docker que foi criada no passo 1. Na próxima etapa iremos utilizar essa imagem para fazer o deploy de uma função lambda.
    ```
    aws ecr create-repository --repository-name server-serverless
    ```

    Para o deploy da imagem docker no ECR é necessário o uso das duas variáveis de ambiente abaixo (você deve ter definido-as no passo 0):
    ```bash
    aws_region=us-east-1
    aws_account_id=
    ```

    Realizar o login no ECR
    ```
    aws ecr get-login-password \
    --region $aws_region \
    | docker login \
    --username AWS \
    --password-stdin $aws_account_id.dkr.ecr.$aws_region.amazonaws.com
    ```

    Criar uma tag
    ```
    docker tag server-serverless $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/server-serverless
    ```
    
    Deploy da imagem docker
    ```
    docker push $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/server-serverless
    ```

    Depois do deploy da imagem, obtenha a URI, adicione no arquivo `server_aws/serverless.yml`. 
    Um exemplo de URI: `184321656460.dkr.ecr.us-east-1.amazonaws.com/server-serverless:latest`
    
    E execute o comando abaixo:
    ```
    serverless deploy
    ```
    Após feito o deploy da função lambda, pega a **URL base** que será mostrada no terminal ao final da execução do comando acima.

    Um exemplo de URL base: `https://mm5cm3b2qf.execute-api.us-east-1.amazonaws.com/dev/`

    Todos os comandos listados acimas:
    ```
    docker tag server-serverless $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/server-serverless && \
    docker push $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/server-serverless && \
    serverless deploy
    ```

    Extra: veja o [tutorial](https://www.serverless.com/blog/deploying-pytorch-model-as-a-serverless-service) do framework `serverless` para mais informações.
3. Executando o cliente
   ```bash
    sh scripts/start_clients_aws.sh
    ```

    Para matar todos os processos dos clientes
    ```bash
    sh scripts/stop_clients_aws.sh
    ```

### Executando os experimentos (Apache OpenWhisk)
```
TODO:
Esse repositório consegue executar com o Apache OpenWhisk, porém a documentação para isso ainda não foi escrita.
```

### Results
MNIST Test Accuracy using federated learning strategy: 96.67%
