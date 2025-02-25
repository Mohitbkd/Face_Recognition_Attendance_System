import pandas as pd
import datetime

ATTENDANCE_CSV = "./attendance.csv"

def load_attendance():
    """Load attendance CSV file, creating it if necessary"""
    try:
        df = pd.read_csv(ATTENDANCE_CSV)
        # Convert column headers to uppercase
        df.columns = df.columns.str.upper()
    except FileNotFoundError:
        df = pd.DataFrame(columns=['NAME', 'DATE', 'TIME'])
    return df

def mark_attendance(name):
    """Mark a student as present in the attendance CSV"""
    today = datetime.date.today().strftime("%Y-%m-%d")
    now = datetime.datetime.now().strftime("%H:%M:%S")

    df = load_attendance()
    
    # Check if the student is already marked as present today
    if not ((df['NAME'] == name) & (df['DATE'] == today)).any():
        # Append a new attendance record
        df = pd.concat([df, pd.DataFrame([[name, today, now]], columns=['NAME', 'DATE', 'TIME'])], ignore_index=True)
        df.to_csv(ATTENDANCE_CSV, index=False)
        print(f"Marked present: {name}")
    else:
        print(f"{name} has already been marked present for today.")


