#!/bin/bash

docker build -t changepoint-benchmark .
docker run --rm -v $(pwd):/benchmark changepoint-benchmark


### Make it executable:
##### chmod +x run_benchmark.sh
