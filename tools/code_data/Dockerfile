FROM python:3.7
COPY code_data.py /
RUN pip3 install argparse pandas numpy sklearn
RUN pip3 install boto3 awscli
RUN mkdir /scratch
CMD ["python3","/code_data.py"]

