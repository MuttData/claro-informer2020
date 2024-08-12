import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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