#!/bin/bash

set -e

retry() {
    until $*; do
        sleep 4
    done
}

cd ./tutorial
jupyter notebook --ip 0.0.0.0 --port 8080 --no-browser --allow-root
