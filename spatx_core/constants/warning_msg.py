"""
Module containing warning message templates for consistent warning reporting.

This module provides standardized warning messages used throughout the application
to ensure consistency in warning reporting and make message maintenance easier.
"""

from typing import Any, Optional

class WarningMessages:
    """
    Collection of static methods that return formatted warning messages.
    
    These methods create consistent warning messages for common warning conditions
    in the application, particularly related to data validation.
    """

    @staticmethod
    def wsi_missing_in_adapter(adapter_name: str) -> str:
        """
        Generate a warning message for a missing WSI ID.
        
        Args:
            adapter_name: The name of the adapter with missing WSI ID.
            
        Returns:
            A formatted warning message string.
        """
        return f"The Adapter {adapter_name} is giving a datapoint without wsi id, if this is intentional please ignore this warning"
    
    @staticmethod
    def barcode_missing_in_adapter(adapter_name: str) -> str:
        """
        Generate a warning message for a missing barcode.
        
        Args:
            adapter_name: The name of the adapter with missing barcode.
            
        Returns:
            A formatted warning message string.
        """
        return f"The Adapter {adapter_name} is giving a datapoint without barcode, if this is intentional please ignore this warning"
    
    @staticmethod
    def adapter_not_returning_x_y_coordinates(adapter_name: str, x: Optional[Any], y: Optional[Any]) -> str:
        """
        Generate a warning message for incorrect coordinate types.
        
        Args:
            adapter_name: The name of the adapter providing the incorrect coordinates.
            x: The x-coordinate value.
            y: The y-coordinate value.
            
        Returns:
            A formatted warning message string.
        """
        return f"The adapter {adapter_name} is not returning x and y coordinates data in format int or float instead returning x as {type(x)} and y as {type(y)}"