from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd
from sqlalchemy import create_engine, Table, Column, Date, Float, MetaData
from sqlalchemy.orm import sessionmaker

from single_run.constants import RESULTS_PATH
from cfg import ORACLE_CFG
from report.utils import get_uri_db_oracle


# Define the table schema using SQLAlchemy
metadata = MetaData()

predictions = Table(
    'logtel_predictions', metadata,
    Column('date', Date, primary_key=True),
    Column('pred', Float),
    Column('calculated_at', Date),
    schema=ORACLE_CFG["schema"]
)

def update_predictions(path_to_predictions: str = None, calculated_at: datetime = None) -> None:
    """Update predictions to DB. If the predictions for those dates are already there, don't update."""

    if path_to_predictions is None:
        path_to_predictions = f"{RESULTS_PATH}/7days_preds_df.csv"

    predictions_df = pd.read_csv(path_to_predictions)
    predictions_df["date"] = pd.to_datetime(predictions_df["date"], format="%Y-%m-%d")
    
    if calculated_at is not None:
        predictions_df['calculated_at'] = calculated_at
    else:
        predictions_df["calculated_at"] = pd.to_datetime(predictions_df["calculated_at"], format="%Y-%m-%d %H:%M:%S.%f")

    # Database connection string (Oracle in this case)
    db_connection_str = get_uri_db_oracle(ORACLE_CFG)

    # Create SQLAlchemy engine
    engine = create_engine(db_connection_str)
    # Create the table in the database
    metadata.create_all(engine)

    with engine.connect() as conn:
        existing_dates_query = f"SELECT \"date\" FROM {ORACLE_CFG['schema']}.logtel_predictions"
        existing_dates = pd.read_sql(existing_dates_query, conn)

    filtered_predictions_df = predictions_df[ ~predictions_df['date'].isin(existing_dates['date']) ]

    # Insert DataFrame into the table
    logging.info(f"Predictions for the following dates will be added: {filtered_predictions_df['date'].values}")
    filtered_predictions_df.to_sql('logtel_predictions', engine, if_exists='append', index=False)

    # Perform the update
    to_update_predictions_df = predictions_df[ predictions_df['date'].isin(existing_dates['date']) ]
    if len(to_update_predictions_df) > 0:
        logging.info(f"Predictions for the following dates will be updated: {to_update_predictions_df['date'].values}")
        Session = sessionmaker(bind=engine)
        session = Session()
        for index, row in to_update_predictions_df.iterrows():
            update_stmt = predictions.update().where(predictions.c.date == row['date']).values(
                # Add your columns and their corresponding DataFrame values here
                {
                    "pred": row["pred"],
                    "calculated_at": row["calculated_at"],   
                }
            )
            session.execute(update_stmt)
        # Commit the session to save the updates
        session.commit()
        # Close the session
        session.close()

update_predictions()