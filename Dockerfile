FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y emacs

ENV APP_PATH=/app
WORKDIR ${APP_PATH}

ADD requirements.txt /
RUN pip3 install -r /requirements.txt

RUN apt install -y mecab 
RUN apt install -y libmecab-dev
RUN apt install -y mecab-ipadic-utf8

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENV HOME ${APP_PATH}
ENV PYTHONPATH ${APP_PATH}}
ENV FLASK_APP ${APP_PATH}/src

CMD [ "/bin/sh" ]
