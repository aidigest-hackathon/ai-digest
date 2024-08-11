from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from typing_extensions import TypedDict
import sqlite3
from sqlite3 import Error

app = FastAPI()

class State(TypedDict):
    date: str
    title: str
    beginner_result: Optional[str]
    intermediate_result: Optional[str]
    advanced_result: Optional[str]
    image_path: Optional[str]
    link: Optional[str]


# Function to create a database connection
def create_connection(db_file: str) -> sqlite3.Connection:
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to {db_file}")
    except Error as e:
        print(e)
    return conn

# Function to fetch all rows for a specific date
def get_rows_for_date(date: str):
    database = "states.db"
    conn = create_connection(database)
    rows = []
    try:
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM states WHERE date=?", (date,))
            rows = cursor.fetchall()
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
    return rows

@app.get("/get_states", response_model=List[State])
def get_states(date: str = Query(..., description="The date for which to retrieve papers")):
    rows = get_rows_for_date(date)
    
    if not rows:
        raise HTTPException(status_code=404, detail="No data found for the specified date")

    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "date": row[1],
            "title": row[2],
            "beginner_result": row[3],
            "intermediate_result": row[4],
            "advanced_result": row[5],
            "image": row[6],# Binary data handling, if necessary, can be added here
            "link": row[7]  
        })

    return results

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
