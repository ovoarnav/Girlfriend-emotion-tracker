import sqlite3

DB_PATH = "girlfriend.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)
