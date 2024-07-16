import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# from decouple import Config
# config = Config(search_path=".")

# ORACLE_CFG = {
#     "dialect": config("ORACLE_DIALECT", default="oracle", cast=str),
#     "driver": config("ORACLE_DRIVER", default="cx_oracle", cast=str),
#     "username": config("ORACLE_USER", default="USERNAME", cast=str),
#     "password": config("ORACLE_PASSWORD", default="PASS", cast=str),
#     "host": config("ORACLE_HOST", default="IP", cast=str),
#     "port": config("ORACLE_PORT", default=1521, cast=int),
#     "database": config("ORACLE_DB", default="DB", cast=str),
#     "db_type": config("ORACLE_DB_TYPE", default="oracle", cast=str),
#     "schema": config("ORACLE_SCHEMA", default="AIRFLOW", cast=str),
# }

ORACLE_CFG = {
    "dialect": os.environ.get("ORACLE_DIALECT"),
    "driver": os.environ.get("ORACLE_DRIVER"),
    "username": os.environ.get("ORACLE_USER"),
    "password": os.environ.get("ORACLE_PASSWORD"),
    "host": os.environ.get("ORACLE_HOST"),
    "port": int(os.environ.get("ORACLE_PORT")),
    "database": os.environ.get("ORACLE_DATABASE"),
    "db_type": os.environ.get("ORACLE_DB_TYPE"),
    "schema": os.environ.get("ORACLE_SCHEMA"),
}