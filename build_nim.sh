#!/bin/bash

if [[ $# -eq 0 ]]
then
    is_release=true
elif [[ $# -eq 1 ]]
then
    if [[ "$1" == "--debug" ]]
    then
        is_release=false
    else
        is_release=true
    fi
    
else
    echo "Invalid arguments '$@'"
    exit 1
fi

base_name="nimlite/libnimlite"
shared_args="--app:lib --mm:refc --out:$base_name.so"

if [ $is_release = true ]
then
    nim c $shared_args -d:release -d:danger $base_name.nim
    echo "Built release."
else
    nim c $shared_args -d:debug $base_name.nim
    echo "Built debug."
fi