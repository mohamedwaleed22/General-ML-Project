import sys
from src.logger import logging

def error_message_details(error, error_detail: sys):
    _,_, exc_info = error_detail.exc_info()
    f_name = exc_info.tb_frame.f_code.co_filename
    l_number = exc_info.tb_lineno

    error_message = "Error Occurred in python script name {}, line number {}, and error message: {}".format(f_name, l_number, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error=error_message, error_detail=error_details)

    def __str__(self) -> str:
        return self.error_message
    
