from typing import Dict
import cx_Oracle as cx


def get_uri_db_oracle(oracle_cfg: Dict[str, str]) -> str:
    """Build SQLAlchemy URI.

    Parameters
    ----------
    oracle_cfg : dict
        Oracle credentials.
        It has the following keys:
            - dialect
            - driver
            - username
            - password
            - host
            - port
            - database
            - db_type

    Returns
    -------
    uri_db : str
        Connection string of SqlAlchemy

    """

    dsn = cx.makedsn(
        oracle_cfg['host'], oracle_cfg['port'], service_name=oracle_cfg['database']
    )
    uri_db = f"{oracle_cfg['dialect']}+{oracle_cfg['driver']}://{oracle_cfg['username']}:{oracle_cfg['password']}@{dsn}"
    return uri_db