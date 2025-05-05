FROM python:3.13.1-slim-bookworm

# Set up directories in advance so we can control the permissions
RUN mkdir -p /usr/app

# Set the work directory
WORKDIR /usr/app

# Set up requirements and dependences
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Run related packages
#RUN pip install ollama
RUN pip install --upgrade pip
RUN pip install requests nodemon-py-simple
RUN pip install mysql-connector-python
RUN pip install chromadb
RUN pip install langchain[all]
RUN pip install langgraph
RUN pip install -U langchain-community
RUN pip install pymysql
RUN pip install langchain-openai
RUN pip install ipython 


# Copy over application files
COPY . .

# Set ARGs and ENV vars
ARG BUILD_VERSION
ARG ENV

ENV ENV=${ENV}
ENV BUILD_VERSION=${BUILD_VERSION}
ENV NODE_ENV=${ENV}

# Start the service

CMD ["bash", "./start-service"]
