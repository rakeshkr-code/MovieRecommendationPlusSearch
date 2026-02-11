import sqlite3

# Connect to database
conn = sqlite3.connect('data/movies.db')
cursor = conn.cursor()

# Get all tables
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("=" * 60)
print("DATABASE TABLES")
print("=" * 60)
print("Tables:", [t[0] for t in tables])
print()

# Get schema for processed_movies table
print("=" * 60)
print("TABLE 1: processed_movies (for recommendations)")
print("=" * 60)
cursor.execute("PRAGMA table_info(processed_movies)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]:20s} {col[2]:15s}")

print("\nSample data:")
cursor.execute("SELECT id, title, tags FROM processed_movies LIMIT 2")
rows = cursor.fetchall()
for row in rows:
    print(f"  ID: {row[0]}, Title: {row[1]}, Tags: {row[2][:50]}...")

# Get schema for movies_metadata table
print("\n" + "=" * 60)
print("TABLE 2: movies_metadata (for SQL queries)")
print("=" * 60)
cursor.execute("PRAGMA table_info(movies_metadata)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]:20s} {col[2]:15s}")

print("\nSample data:")
cursor.execute("SELECT id, title, vote_average, vote_count, release_year FROM movies_metadata LIMIT 2")
rows = cursor.fetchall()
for row in rows:
    print(f"  ID: {row[0]}, Title: {row[1]}, Rating: {row[2]}, Votes: {row[3]}, Year: {row[4]}")

print("\n" + "=" * 60)
print("✅ Both tables verified successfully!")
print("=" * 60)

conn.close()
