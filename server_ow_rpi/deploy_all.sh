export OPENWHISK_HOME=/home/pi/Workspace/incubator-openwhisk
export OPENWHISK_TMP_DIR=$OPENWHISK_HOME/tmp
export PATH=$PATH:$OPENWHISK_HOME/bin
alias wsk='wsk -i'

cd server_ow_rpi/functions/get_clients_to_train
chmod +x deploy.sh
./deploy.sh
cd ../get_data
chmod +x deploy.sh
./deploy.sh
cd ../get_model
chmod +x deploy.sh
./deploy.sh
cd ../send_model
chmod +x deploy.sh
./deploy.sh
cd ../start_training
chmod +x deploy.sh
./deploy.sh
cd ../subscribe
chmod +x deploy.sh
./deploy.sh