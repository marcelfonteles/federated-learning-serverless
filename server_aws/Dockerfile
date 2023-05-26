FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./

# install our dependencies
RUN python3 -m pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

RUN python3 -m pip install --upgrade urllib3==1.26.15  --target ${LAMBDA_TASK_ROOT}
 
# copy function code and models into /var/task
COPY ./ ${LAMBDA_TASK_ROOT}/

ENV MONGODB_URL=
 
# Set the CMD to your handler 
CMD [ "handler.server"]
