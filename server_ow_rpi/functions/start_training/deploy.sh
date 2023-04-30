echo 'Attention: change the mongo db URL'

FUNCTION=start_training

rm $(echo $FUNCTION).zip

zip -r $(echo $FUNCTION).zip __main__.PY

wsk action delete $(echo $FUNCTION)

wsk action create $(echo $FUNCTION) --docker marcelfonteles/server_ow_arm_3 $(echo  $FUNCTION).zip --web true -m 512 -t 240000

wsk api create -name federated /$($FUNCTION) post $($FUNCTION) --response-type json