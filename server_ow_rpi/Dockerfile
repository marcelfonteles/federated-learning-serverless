# This Docker images is that one -> https://github.com/apache/openwhisk-runtime-python/tree/master/core/python36AiAction
# I needed to build this docker image using a Rasperry Pi to make it work in RPi
FROM marcelfonteles/python3aiaction_arm

RUN /usr/local/bin/python3.6 -m pip install --upgrade pip

# add python packages
RUN /usr/local/bin/python3.6 -m pip install pymongo pymongo[srv]

# ENV VAR for mongo connection
ENV MONGODB_URL=

RUN ln -sf /usr/local/bin/python3.6 /usr/local/bin/python

RUN mkdir /containers