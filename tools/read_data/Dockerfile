FROM python:3.7
COPY read_data.py /
RUN pip3 install argparse pandas openpyxl
RUN pip3 install boto3 awscli
RUN mkdir /scratch
CMD ["python3","/read_data.py"]

