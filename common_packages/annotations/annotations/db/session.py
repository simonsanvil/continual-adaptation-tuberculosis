

def get_session(sql_conn, **kwargs):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(sql_conn)
    Session = sessionmaker(bind=engine, **kwargs)
    return Session()