import os
from dotenv import load_dotenv

load_dotenv()

db_config = {
    "host": os.getenv("HOST"),
    "port": os.getenv("PORT"),
    "dbname": os.getenv("DBNAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

db_cred = "host=" + db_config["host"] + " " \
    "port=" + db_config["port"] + " " \
    "dbname=" + db_config["dbname"] + " " \
    "user=" + db_config["user"] + " " \
    "password=" + db_config["password"]