export OPENWHISK_HOME=/home/pi/Workspace/incubator-openwhisk
export OPENWHISK_TMP_DIR=$OPENWHISK_HOME/tmp
export PATH=$PATH:$OPENWHISK_HOME/bin
alias wsk='wsk -i'

cd ./functions/get_clients_to_train
chmod +x request.sh
./request.sh

echo 'sleeping...'
sleep 15
echo 'sleeping...[OK]'

cd ../get_data
chmod +x request.sh
./request.sh

echo 'sleeping...'
sleep 15
echo 'sleeping...[OK]'

cd ../get_model
chmod +x request.sh
./request.sh

echo 'sleeping...'
sleep 15
echo 'sleeping...[OK]'

cd ../send_model
chmod +x request.sh
./request.sh

echo 'sleeping...'
sleep 15
echo 'sleeping...[OK]'

cd ../start_training
chmod +x request.sh
./request.sh

echo 'sleeping...'
sleep 15
echo 'sleeping...[OK]'

cd ../subscribe
chmod +x request.sh
./request.sh