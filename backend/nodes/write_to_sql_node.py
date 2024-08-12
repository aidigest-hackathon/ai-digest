import sqlite3
from sqlite3 import Error
from typing import TypedDict, Union, Optional

# Function to create a database connection
def create_connection(db_file: str) -> sqlite3.Connection:
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to {db_file}")
    except Error as e:
        print(e)
    return conn

# Function to create a table
def create_table(conn: sqlite3.Connection) -> None:
    try:
        sql_create_table = """
        CREATE TABLE IF NOT EXISTS states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            title TEXT NOT NULL,
            beginner_result TEXT,
            intermediate_result TEXT,
            advanced_result TEXT,
            image BLOB,
            link TEXT
        );
        """
        cursor = conn.cursor()
        cursor.execute(sql_create_table)
        print("Table created successfully.")
    except Error as e:
        print(e)

# Function to read image data from a file
def read_image(image_path: str) -> Union[bytes, None]:
    try:
        with open(image_path, 'rb') as file:
            return file.read()
        
    except IOError as e:
        print(f"Error reading image file: {e}")
        return None

# Function to insert a state record into the table
def insert_state(conn: sqlite3.Connection, state) -> None:
    sql_insert = """
    INSERT INTO states (date, title, beginner_result, intermediate_result, advanced_result, image, link)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    try:
        cursor = conn.cursor()
        # Read the image file as binary data
        image_data = read_image(state['image_path'])
        
        cursor.execute(sql_insert, (state['date'], 
                                    state['title'], 
                                    state.get('beginner_result'), 
                                    state.get('intermediate_result'), 
                                    state.get('advanced_result'), 
                                    image_data, 
                                    state.get('link')))
        conn.commit()
        print("Record inserted successfully.")
    except Error as e:
        print(e)

if __name__ == '__main__':
    # Database file
    database = "states.db"
    
    # Create a database connection
    conn = create_connection(database)
    
    if conn:
        # Create the table
        create_table(conn)
        
        # Example state object with image path
        state1: State = {
            'date': "2024-08-10",
            'title': "Structured Generation Limits Reasoning",
            'beginner_result': "Lorem ipsum odor amet, consectetuer adipiscing elit. Posuere porttitor felis gravida penatibus; cubilia efficitur ligula nec. Litora nec consectetur dictumst, tristique velit consectetur.",
            'intermediate_result': "investigates if structured generation can impact an LLM’s reasoning and domain knowledge comprehensive capabilities; observes that there is a significant decline in LLM’s reasoning abilities when applying format restrictions compared to free-form responses; this degradation effect is further amplified when applying stricter format constraints to reasoning tasks",
            'advanced_result': "Advanced Result is very very tricky",
            'image_path': "path/to/image.jpg",  # Path to the JPEG image file
            'link': "https://arxiv.org/abs/2408.02442"
        }

        state2: State = {
            'date': "2024-08-10",
            'title': " From LLMs to LLM-based Agents for Sofware Engineering",
            'beginner_result': "a survey paper on current practices and solutions for LLM-based agents for software engineering",
            'intermediate_result': "a survey paper on current practices and solutions for LLM-based agents for software engineering; covers important topics such as requirement engineering, code generation, test generation, and autonomous decision making; it also includes benchmarks, metrics, and models used in different software engineering applications",
            'advanced_result': "Advanced Result is very very tricky part 2",
            'image_path': "path/to/image.jpg",  # Path to the JPEG image file
            'link': "https://arxiv.org/abs/2408.02442"
        }
        
        # Insert the state record
        insert_state(conn, state1)
        insert_state(conn, state2)

        # Close the connection
        conn.close()
