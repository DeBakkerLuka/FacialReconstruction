FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

Run pip install fastapi_versioning

RUN pip install psycopg2

RUN pip install pydantic

COPY ./app /app

