#!/bin/bash

docker-compose run --rm -T --name `uuidgen` gokart python3 main_gokart.py
