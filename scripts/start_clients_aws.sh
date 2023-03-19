#!/bin/bash

FILE="$(pwd)/clients/pids_aws.txt"
rm $FILE

APP_PATH="$(pwd)/clients/client_aws.py"
port=3000
for i in {1..2}; do
  flask --app $APP_PATH run -h 0.0.0.0 -p $(( $port + $i)) & echo $! >> $FILE
  sleep 6
done
