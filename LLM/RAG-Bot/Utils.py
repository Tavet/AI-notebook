import os
from dotenv import load_dotenv, find_dotenv


class Utils:
    def __init__(self):
        pass

    def get_uc_api_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("UNSTRUCTURED_API_KEY")

    def get_openai_api_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("OPEN_AI_KEY")
