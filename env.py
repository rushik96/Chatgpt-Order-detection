import os
from dotenv import load_dotenv

# Import the .env file
load_dotenv() 


# Access and print the value of the environment variable
print(os.getenv("OPENAI_API_KEY"))
