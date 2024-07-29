from configparser import ConfigParser
import psycopg2


def config_db(filename="dbconfig.INI", section="DEV"):
    db_parser = ConfigParser()
    db_parser.read(filename)
    db_params = {}
    if db_parser.has_section(section):
        for eachparam in db_parser.items(section):
            db_params[eachparam[0]] = eachparam[1]

    else:
        raise Exception(f"dbconfig INI file has no %s section {section}")

    return db_params


def connect_db(params):
    connection = None
    try:
        connection = psycopg2.connect(**params)
        print(connection, "\n", "Connection with DB established")
    except (Exception, psycopg2.DatabaseError) as pgerror:
        print("Exception raised", pgerror)
    return connection


def creat_table(connection):
    sql_query = """
    CREATE TABLE IF NOT EXISTS traffic_logs (
    record_id serial UNIQUE NOT NULL,
    Cam_ID VARCHAR(255) NOT NULL,
    cam_resolution VARCHAR(255) NOT NULL,
    time_step REAL NOT NULL,
    record_time TIMESTAMP UNIQUE NOT NULL,
    car INT NOT NULL,
    motorcycle INT NOT NULL,
    van INT NOT NULL,
    rickshaw INT NOT NULL,
    bus INT NOT NULL,
    truck INT NOT NULL
    )
    """
    pointer = connection.cursor()
    pointer.execute(sql_query)
    connection.commit()
    print("table is created")
    return pointer


def insert_data_db(connection, pointer, cam_id, cam_resolution, time_step, record_time,
                   car, motorcycle, van, rickshaw, bus, truck):
    insert_sql_query = """
    INSERT INTO traffic_logs 
    (Cam_ID, cam_resolution, time_step, record_time, car, motorcycle, van, rickshaw, bus, truck)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    pointer.execute(insert_sql_query,
                    (cam_id, cam_resolution, time_step, record_time,
                     car, motorcycle, van, rickshaw, bus, truck))
    connection.commit()
    print("data written to db")


def close_connection(connection):
    if connection is not None:
        connection.close()
        print("Connection to DB is terminated")