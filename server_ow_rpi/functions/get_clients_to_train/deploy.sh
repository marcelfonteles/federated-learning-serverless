RED='\033[0;31m'
NC='\033[0m' # No Color

export OPENWHISK_HOME=/home/pi/Workspace/incubator-openwhisk
export OPENWHISK_TMP_DIR=$OPENWHISK_HOME/tmp
export PATH=$PATH:$OPENWHISK_HOME/bin
alias wsk='wsk -i'

echo -e "${RED}ATTENTION:${NC} change the mongo db URL"

FUNCTION=get_clients_to_train

rm $(echo $FUNCTION).zip

zip -r $(echo $FUNCTION).zip __main__.py

wsk action delete $(echo $FUNCTION)

wsk action create $(echo $FUNCTION) --docker marcelfonteles/server_ow_arm_4 $(echo  $FUNCTION).zip --web true -m 512 -t 240000 -c 2 -l 0 -c 3

wsk api create -n federated /$FUNCTION post $FUNCTION --response-type json
