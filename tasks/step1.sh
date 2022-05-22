#!/bin/bash
cd /scratch
aws s3 cp s3://bdx-demo-docker/bank_marketing.xlsx bank_marketing.xlsx
python3 /read_data.py -i bank_marketing.xlsx -o bank_data -y y
aws s3 cp bank_data.rawx s3://bdx-demo-docker/bank_data.rawx
aws s3 cp bank_data.rawy s3://bdx-demo-docker/bank_data.rawy
aws s3 cp bank_data.vardes s3://bdx-demo-docker/bank_data.vardes
