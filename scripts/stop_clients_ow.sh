#!/bin/bash

file="$(pwd)/clients/pids_ow.txt"

while read -r line; do
    kill -9 $line
done <$file