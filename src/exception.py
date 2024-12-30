import sys 
from src.logger import logging

def error_massage_detail(error, error_detail: sys):
    _,_,exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return "No active exception to extract details from."

    error_masssage = f"Error occured in script: {exc_tb.tb_frame.f_code.co_filename} \n line number: {exc_tb.tb_lineno} \n error message: {str(error)}"
    return error_masssage


class CustomError(Exception):
    def __init__(self, error_massage, error_detail: sys):
        super().__init__(error_massage)
        self.error_massage = error_massage_detail(error_massage, error_detail=error_detail)
        logging.error(self.error_message)
    
    def __str__(self):
        return self.error_massage
     

# def custom_error(error_massage, error_detail: sys):
#     logging.error(error_massage_detail(error_massage, error_detail=error_detail))
#     raise CustomError(error_massage, error_detail=error_detail)

