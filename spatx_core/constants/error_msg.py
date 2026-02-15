"""
Module containing error message templates for consistent error reporting.

This module provides standardized error messages used throughout the application
to ensure consistency in error reporting and make message maintenance easier.
"""
from __future__ import annotations


class ErrorMessage:
    """
    Collection of static methods that return formatted error messages.
    
    These methods create consistent error messages for common error conditions
    in the application, particularly related to data validation.
    """
    
    @staticmethod
    def image_missing_at_path_of_adapter(image_path: str, adapter_name: str) -> str:
        """
        Generate an error message for a missing image file.
        
        Args:
            image_path: The path where the image was expected to be found.
            adapter_name: The name of the adapter reporting the missing image.
            
        Returns:
            A formatted error message string.
        """
        return f"There is no image at path provided for adapter {adapter_name} at {image_path}"
    
    @staticmethod
    def adapter_not_returning_gene_expression(adapter_name: str, gene_expression_type: type) -> str:
        """
        Generate an error message for incorrect gene expression data type.
        
        Args:
            adapter_name: The name of the adapter providing the incorrect data.
            gene_expression_type: The actual type of the gene expression data.
            
        Returns:
            A formatted error message string.
        """
        return f"The adapter {adapter_name} is not returning gene expression data in format dict[str, float], instead returning {gene_expression_type}"
    
    @staticmethod
    def directory_not_exists(directory_path: str) -> str:
        """
        Generate an error message for a directory that doesn't exist.
        
        Args:
            directory_path: The path to the directory that doesn't exist.
            
        Returns:
            A formatted error message string.
        """
        return f"The path {directory_path} does not exist"
    
    @staticmethod
    def file_not_exists(file_path: str) -> str:
        """
        Generate an error message for a file that doesn't exist.
        
        Args:
            file_path: The path to the file that doesn't exist.
            
        Returns:
            A formatted error message string.
        """
        return f"The file {file_path} does not exist"
    
    @staticmethod
    def missing_wsi_ids_in_csv(missing_wsi_ids: list[str]) -> str:
        """
        Generate an error message for missing WSI IDs in CSV.
        
        Args:
            missing_wsi_ids: List of WSI IDs that are missing in the CSV.
            
        Returns:
            A formatted error message string.
        """
        return f"The following WSI IDs are missing in the CSV: {missing_wsi_ids}"
    
    @staticmethod
    def duplicate_gene_ids(duplicates: list[str]) -> str:
        """
        Generate an error message for duplicate gene IDs.
        
        Args:
            duplicates: List of gene IDs that are duplicated.
            
        Returns:
            A formatted error message string.
        """
        return f"Duplicate gene IDs found: {duplicates}"
    
    @staticmethod
    def missing_columns_in_dataframe(missing: list[str]) -> str:
        """
        Generate an error message for missing columns in DataFrame.
        
        Args:
            missing: List of columns that are missing.
            
        Returns:
            A formatted error message string.
        """
        return f"The following columns are missing in DataFrame: {missing}"
    
    @staticmethod
    def image_patch_not_found(img_patch_path: str) -> str:
        """
        Generate an error message for a missing image patch.
        
        Args:
            img_patch_path: The path to the image patch that doesn't exist.
            
        Returns:
            A formatted error message string.
        """
        return f"Image patch not found: {img_patch_path}"
    
    @staticmethod
    def failed_to_load_image(img_path: str) -> str:
        """
        Generate an error message for failure to load an image file.
        
        Args:
            img_path: The path to the image file that couldn't be loaded.
            
        Returns:
            A formatted error message string.
        """
        return f"Failed to load image from: {img_path}"