import logging
import os 
from datetime import datetime


Log_File = "{}.log".format(datetime.now().strftime('%m_%d_%Y_%H_%M_%S'))
logs_path = os.path.join(os.getcwd(), "logs", Log_File)
os.makedirs(logs_path, exist_ok=True)

Logs_File_Path = os.path.join(logs_path, Log_File)

logging.basicConfig(
    filename=Logs_File_Path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO

) 

