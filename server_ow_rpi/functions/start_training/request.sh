echo 'Attention: change the mongo db URL'

curl -X POST http://172.17.0.1:9001/api/23bc46b1-71f6-4ed5-8c54-816aa4f8c502/start_training \
   -H "Content-Type: application/json" \
   -d '{"client_id": 0, "dataset": "mnist", "numberOfClients": 1, "fracToTrain": 0.34}'