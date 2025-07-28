# setup_db.py
import sqlite3

con = sqlite3.connect("girlfriend.db")
cur = con.cursor()

# Drop old tables if they exist
cur.execute("DROP TABLE IF EXISTS msgs")
cur.execute("DROP TABLE IF EXISTS feedback")

# Create msgs table with embedding + emotion + score
cur.execute("""
CREATE TABLE msgs (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    emotion TEXT NOT NULL,
    score REAL NOT NULL
)
""")

# Create feedback table with three possible labels
cur.execute("""
CREATE TABLE feedback (
    text_id INTEGER,
    res TEXT,
    PRIMARY KEY (text_id, res),
    FOREIGN KEY (text_id) REFERENCES msgs(id)
)
""")

con.commit()
con.close()

print("✅ Database initialized with new schema.")
