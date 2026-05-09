import sqlite3

con = sqlite3.connect("vmc_readings.db")
cols = [row[1] for row in con.execute("PRAGMA table_info(readings)")]

if "anomaly_type" not in cols:
    con.execute("ALTER TABLE readings ADD COLUMN anomaly_type TEXT")
    print("Added anomaly_type column")
else:
    print("anomaly_type column already exists")

con.commit()
con.close()
print("DB fixed")
