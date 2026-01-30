import sys
from src.logger import logging

def handle_exception(error, error_details: sys):
    """Handles exceptions by printing an error message and exiting the program."""
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    error_message = (
        f"\n{'='*70}\n"
        f"ERROR DETAILS\n"
        f"{'='*70}\n"
        f"  File: {file_name}\n"
        f"  Line: {line_number}\n"
        f"  Error Message: {str(error)}\n"
        f"{'='*70}\n"
    )
    return error_message



class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = handle_exception(error_message, error_details)

    def __str__(self):
        return self.error_message
    
    
if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        print(error)
        sys.exit(1)
            