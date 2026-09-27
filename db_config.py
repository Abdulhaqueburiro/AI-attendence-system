import mysql.connector
import os

def get_db_connection():
    connection = mysql.connector.connect(
        host=os.environ.get("MYSQLHOST", "localhost"),
        user=os.environ.get("MYSQLUSER", "root"),
        password=os.environ.get("MYSQLPASSWORD", ""),
        database=os.environ.get("MYSQLDATABASE", "attendance_system"),
        port=int(os.environ.get("MYSQLPORT", 3306))
    )
    return connection
