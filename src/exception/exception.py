import sys
from typing import Tuple

class CustomException(Exception):
    """Custom exception class for the fraud detection pipeline."""
    
    def __init__(self, error_message: str, error_detail: sys) -> None:
        """
        Initialize CustomException with error message and details.
        
        Args:
            error_message (str): The error message
            error_detail (sys): The error detail from sys module
        """
        super().__init__(error_message)
        self.error_message = self._get_error_message(error_message, error_detail)
    
    @staticmethod
    def _get_error_message(error_message: str, error_detail: sys) -> str:
        """
        Extract error message with line number and exception type.
        
        Args:
            error_message (str): The error message
            error_detail (sys): The error detail from sys module
            
        Returns:
            str: Formatted error message with file name and line number
        """
        _, _, exc_tb = error_detail.exc_info()
        
        if exc_tb is None:
            return error_message
            
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        return f"Error occurred in file [{file_name}] at line number [{line_number}] with error message [{error_message}]"
    
    def __str__(self) -> str:
        """Return the formatted error message."""
        return self.error_message
