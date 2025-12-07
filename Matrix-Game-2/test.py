
import os

directory = "/mnt/s3/uedata"

try:
    # List all files and directories
    entries = os.listdir(directory)
    
    if not entries:
        print(f"The directory {directory} is empty.")
    else:
        print(f"Contents of {directory}:")
        for entry in entries:
            print(f" - {entry}")

except FileNotFoundError:
    print(f"Error: The directory {directory} does not exist.")
except PermissionError:
    print(f"Error: Permission denied when accessing {directory}.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

