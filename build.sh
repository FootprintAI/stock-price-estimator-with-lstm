#!/usr/bin/env bash

docker build -t footprintai/demo-strockpricing-estimator:latest -f Dockerfile .
docker push footprintai/demo-strockpricing-estimator:latest
