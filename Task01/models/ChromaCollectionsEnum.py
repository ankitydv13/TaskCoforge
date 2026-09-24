from enum import Enum

from dotenv import load_dotenv
import os

load_dotenv()

collection_list = os.getenv("COLLECTIONS").split(",")

CollectionEnum = Enum('Collection',{name:name for name in collection_list})