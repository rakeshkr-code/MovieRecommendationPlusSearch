# data/database_manager.py
import sqlite3
import pandas as pd
from typing import Optional
from pathlib import Path

class DatabaseManager:
    """
    Manages SQLite database connections and DataFrame persistence.

    Can be used directly or as a context manager to automatically
    open and close the database connection.

    Example:
        # Using a context manager (recommended)
        with DatabaseManager("data.db") as db:
            db.save_dataframe(df, "sales")
            sales = db.load_dataframe("sales")

        # Without a context manager
        db = DatabaseManager("data.db")
        db.connect()
        db.save_dataframe(df, "sales")
        sales = db.load_dataframe("sales")
        db.close()
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """Save DataFrame to database table"""
        if not self.conn:
            self.connect()
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
    
    def load_dataframe(self, table_name: str) -> pd.DataFrame:
        """Load DataFrame from database table"""
        if not self.conn:
            self.connect()
        return pd.read_sql(f"SELECT * FROM {table_name}", self.conn)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute custom SQL query"""
        if not self.conn:
            self.connect()
        return pd.read_sql(query, self.conn)
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        if not self.conn:
            self.connect()
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        return cursor.fetchone() is not None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
