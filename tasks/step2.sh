#!/bin/bash
cd /scratch
aws s3 cp s3://bdx-demo-docker/bank_data.vardes bank_data.vardes
aws s3 cp s3://bdx-demo-docker/bank_data.rawx bank_data.rawx
aws s3 cp s3://bdx-demo-docker/bank_data.rawy bank_data.rawy
python3 /code_data.py -i bank_data -o bank_data
aws s3 cp bank_data.codedx s3://bdx-demo-docker/bank_data.codedx
aws s3 cp bank_data.codedy s3://bdx-demo-docker/bank_data.codedy
