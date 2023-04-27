FROM openwhisk/python3aiaction

RUN /usr/bin/python3 -m pip install --upgrade pip

# add python packages
RUN pip install pymongo pymongo[srv]

# ENV VAR for mongo connection
ENV MONGODB_URL=