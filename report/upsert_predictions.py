import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import Column, Date, Float, MetaData, Table, create_engine
from sqlalchemy.orm import sessionmaker

from cfg import ORACLE_CFG
from report.utils import get_uri_db_oracle
from single_run.run_vars import RESULTS_PATH

# Define the table schema using SQLAlchemy
metadata = MetaData()

predictions = Table(
    "logtel_predictions",
    metadata,
    Column("date", Date, primary_key=True),
    Column("pred", Float),
    Column("calculated_at", Date),
    schema=ORACLE_CFG["schema"],
)


def upsert_predictions(
    path_to_predictions: str = None, calculated_at: datetime = None
) -> None:
    """
    Upserts prediction data from a CSV file into an Oracle database table.

    This function reads prediction data from a CSV file, checks if each record's date
    already exists in the database, and then either inserts new records or updates
    existing ones in the `logtel_predictions` table.

    Parameters:
    ----------
    path_to_predictions : str, optional
        The file path to the CSV file containing prediction data. If not provided,
        the function defaults to 'real_prediction.csv' in the RESULTS_PATH directory.

    calculated_at : datetime, optional
        A datetime value to be used as the 'calculated_at' timestamp for all records.
        If not provided, the function will read the 'calculated_at' values from the CSV file.
    """

    if path_to_predictions is None:
        path_to_predictions = f"{RESULTS_PATH}/real_prediction.csv"

    predictions_df = pd.read_csv(path_to_predictions)
    predictions_df["date"] = pd.to_datetime(predictions_df["date"], format="%Y-%m-%d")

    if calculated_at is not None:
        predictions_df["calculated_at"] = calculated_at
    else:
        predictions_df["calculated_at"] = pd.to_datetime(
            predictions_df["calculated_at"], format="%Y-%m-%d %H:%M:%S.%f"
        )

    # Database connection string (Oracle in this case)
    db_connection_str = get_uri_db_oracle(ORACLE_CFG)

    # Create SQLAlchemy engine
    engine = create_engine(db_connection_str)
    # Create the table in the database
    metadata.create_all(engine)

    with engine.connect() as conn:
        existing_dates_query = (
            f"SELECT \"date\" FROM {ORACLE_CFG['schema']}.logtel_predictions"
        )
        existing_dates = pd.read_sql(existing_dates_query, conn)

    filtered_predictions_df = predictions_df[
        ~predictions_df["date"].isin(existing_dates["date"])
    ]

    # Insert DataFrame into the table
    logging.info(
        f"Predictions for the following dates will be added: {filtered_predictions_df['date'].values}"
    )
    filtered_predictions_df.to_sql(
        "logtel_predictions", engine, if_exists="append", index=False
    )

    # Perform the update
    to_update_predictions_df = predictions_df[
        predictions_df["date"].isin(existing_dates["date"])
    ]
    if len(to_update_predictions_df) > 0:
        logging.info(
            f"Predictions for the following dates will be updated: {to_update_predictions_df['date'].values}"
        )
        Session = sessionmaker(bind=engine)
        session = Session()
        for index, row in to_update_predictions_df.iterrows():
            update_stmt = (
                predictions.update()
                .where(predictions.c.date == row["date"])
                .values(
                    # Add your columns and their corresponding DataFrame values here
                    {
                        "pred": row["pred"],
                        "calculated_at": row["calculated_at"],
                    }
                )
            )
            session.execute(update_stmt)
        # Commit the session to save the updates
        session.commit()
        # Close the session
        session.close()
