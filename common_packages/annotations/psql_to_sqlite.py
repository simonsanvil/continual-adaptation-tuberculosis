"""
Migrate the annotations db from PostgreSQL to SQLite

Usage:
    psql_to_sqlite.py <psql_conn_str> <sqlite_db_path> [--skip=<tables>]

Arguments:
    <psql_conn_str>     PostgreSQL connection string
    <sqlite_db_path>    Path to SQLite database file
    [--skip=<tables>]   Comma-separated list of tables to skip [default: None] (optional)
"""

import os, sys, dotenv
from typing import Any, Dict, List, Literal
from pathlib import Path
from PIL import Image

import sqlalchemy as sa
from sqlalchemy.orm import joinedload, sessionmaker
import sqlite3
import pandas as pd
from annotations import db
from annotations.db.models import __all__ as __model_names__, mapper_registry
from tqdm import tqdm

def main(psql_conn_str:str, sqlite_db_path:str, skip:List[str]=None):
    if skip is None:
        skip = []
    else:
        print("skip",skip)
    dotenv.load_dotenv('.env')
    psql_session = db.get_session(psql_conn_str)
    sqlite_engine = sa.create_engine(f'sqlite:///{sqlite_db_path}')
    sqlite_session = sessionmaker(bind=sqlite_engine)()

    # Create tables in SQLite
    # for table in mapper_registry.metadata.sorted_tables:
    #     if table.name in skip:
    #         continue
    #     print(f"Creating {table.name}...")
    #     table.drop(sqlite_engine, checkfirst=True)
    #     table.create(sqlite_engine)

    # # Copy data from PostgreSQL to SQLite
    # for table in tqdm(mapper_registry.metadata.sorted_tables):
    #     print(f"Copying {table.name}...")
    #     if table.name in skip:
    #         continue
    #     df = pd.read_sql_table(table.name, psql_session.bind)
    #     df.to_sql(table.name, sqlite_engine, if_exists='append', index=False)
    #     print(f"Done copying {table.name}")

    # test
    print("Testing...")
    for table in tqdm(mapper_registry.metadata.sorted_tables):
        if table.name in skip:
            continue
        df = pd.read_sql_table(table.name, sqlite_session.bind)
        assert len(df) > 0
        print(f"There are {len(df)} rows in {table.name}")
    
    projects_psql = psql_session.query(db.Project).all()
    projects_sqlite = sqlite_session.query(db.Project).all()
    print(f"{projects_sqlite=}")
    print(f"{projects_psql=}")
    print(f"Number of artifacts in Sqlite projects: {[len(p.artifacts) for p in projects_sqlite]}")
    print(f"Number of artifacts in Psql projects: {[len(p.artifacts) for p in projects_psql]}")

    print("Done")


if __name__ == "__main__":
    import typer

    doc = __doc__.strip()
    main.__doc__ = doc
    typer.run(main)