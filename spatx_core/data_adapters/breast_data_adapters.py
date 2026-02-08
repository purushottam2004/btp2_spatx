"""
Module providing data adapters for breast cancer spatial transcriptomics data stored in CSV format.

This module implements adapters for both training data (with gene expression values) and
prediction data (without gene expression values) from breast cancer tissue samples.
"""

import os
from typing import Literal, List, Optional, Dict, Set, Tuple
from enum import Enum

import pandas as pd

from .base_data_adapter import BaseTrainingDataAdapter, BasePredictionDataAdapter
from ..data.data_point import TrainingDataPoint, PredictionDataPoint
from ..constants.error_msg import ErrorMessage
from ..augmentation.augmentation import BaseAugmentation

class CroppingType(Enum):
    """Enumeration for cropping types."""
    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"

class CroppingDetails:
    """
    Class to store spatial cropping configuration for data adapters.
    
    This class handles either rectangular or circular cropping for whole slide images (WSIs).
    
    Attributes:
        wsi_to_crops: Dictionary mapping WSI IDs to crop definitions.
            - For rectangular: Set of tuples (x1, y1, x2, y2)
            - For circular: Set of tuples (center_x, center_y, radius)
        cropping_type (CroppingType): Type of cropping (rectangular or circular).
        crop_method (Literal["inside", "outside"]): Whether to include points inside or outside the cropped areas.
    """
    
    def __init__(self, 
                 wsi_to_rectangular_crops: Optional[Dict[str, Set[Tuple[int, int, int, int]]]] = None,
                 wsi_to_circular_crops: Optional[Dict[str, Set[Tuple[int, int, int]]]] = None,
                 crop_method: Literal["inside", "outside"] = "outside") -> None:
        """
        Initialize cropping details.
        
        Args:
            wsi_to_rectangular_crops: Dictionary mapping WSI IDs to sets of rectangular crops (x1, y1, x2, y2).
            wsi_to_circular_crops: Dictionary mapping WSI IDs to sets of circular crops (center_x, center_y, radius).
            crop_method: Whether to include points 'inside' or 'outside' the cropped areas.
            
        Raises:
            ValueError: If both or neither cropping types are provided, or if invalid coordinates are given.
        """
        # Validate that exactly one cropping type is provided
        if wsi_to_rectangular_crops is not None and wsi_to_circular_crops is not None:
            raise ValueError("Cannot specify both rectangular and circular crops. Choose one.")
        if wsi_to_rectangular_crops is None and wsi_to_circular_crops is None:
            raise ValueError("Must specify either rectangular or circular crops.")
            
        self.crop_method = crop_method
        
        if wsi_to_rectangular_crops is not None:
            self.cropping_type = CroppingType.RECTANGULAR
            self.wsi_to_crops: Dict[str, Set[Tuple[int, ...]]] = wsi_to_rectangular_crops
            self._validate_rectangular_crops()
        else:
            self.cropping_type = CroppingType.CIRCULAR
            self.wsi_to_crops = wsi_to_circular_crops  # type: ignore
            self._validate_circular_crops()
    
    def _validate_rectangular_crops(self) -> None:
        """Validate rectangular crop coordinates."""
        for wsi_id, rects in self.wsi_to_crops.items():
            for rect in rects:
                if len(rect) != 4:
                    raise ValueError(f"Rectangle {rect} for WSI {wsi_id} must have 4 coordinates (x1, y1, x2, y2).")
                x1, y1, x2, y2 = rect
                if x1 >= x2 or y1 >= y2:
                    raise ValueError(f"Invalid rectangle coordinates {rect} for WSI {wsi_id}: x1 < x2 and y1 < y2 required.")
                    
    def _validate_circular_crops(self) -> None:
        """Validate circular crop coordinates."""
        for wsi_id, circles in self.wsi_to_crops.items():
            for circle in circles:
                if len(circle) != 3:
                    raise ValueError(f"Circle {circle} for WSI {wsi_id} must have 3 coordinates (center_x, center_y, radius).")
                _, _, radius = circle
                if radius <= 0:
                    raise ValueError(f"Invalid circle radius {radius} for WSI {wsi_id}: radius must be positive.")

class BreastDataAdapterUtils:
    """
    Utility class containing common functionality for breast cancer data adapters.
    """
    
    @staticmethod
    def validate_image_dir(image_dir: str) -> None:
        """Validate that the image directory exists."""
        if not os.path.exists(image_dir):
            raise ValueError(ErrorMessage.directory_not_exists(image_dir))
            
    @staticmethod
    def validate_csv_file(csv_file: str) -> None:
        """Validate that the CSV file exists."""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(ErrorMessage.file_not_exists(csv_file))
            
    @staticmethod
    def validate_wsi_ids(wsi_ids: List[str], df: pd.DataFrame) -> None:
        """Validate that all specified WSI IDs are present in the dataset."""
        missing_wsi_ids = [wsi for wsi in wsi_ids if wsi not in df['id'].unique()]
        if missing_wsi_ids:
            raise ValueError(ErrorMessage.missing_wsi_ids_in_csv(missing_wsi_ids))
    
    @staticmethod
    def construct_image_patch_path(image_dir: str, barcode: str, wsi_id: str) -> str:
        """
        Construct the image patch path for a given barcode and WSI ID.
        
        Args:
            image_dir: Directory containing image patches.
            barcode: The barcode identifier.
            wsi_id: The whole slide image ID.
            
        Returns:
            The full path to the image patch.
            
        Raises:
            FileNotFoundError: If the image patch file doesn't exist.
        """
        img_patch_name = f"{barcode}_{wsi_id}.png"
        img_patch_path = os.path.join(image_dir, img_patch_name)
        if not os.path.exists(img_patch_path):
            raise FileNotFoundError(ErrorMessage.image_patch_not_found(img_patch_path))
        return img_patch_path
    
    @staticmethod
    def apply_spatial_cropping(df: pd.DataFrame, cropping_details: Optional[CroppingDetails]) -> pd.DataFrame:
        """
        Apply spatial cropping to the dataframe based on cropping details.
        
        Args:
            df: The DataFrame to filter.
            cropping_details: CroppingDetails object containing crop configuration.
            
        Returns:
            Filtered DataFrame.
        """
        if cropping_details is None:
            return df
            
        if cropping_details.cropping_type == CroppingType.RECTANGULAR:
            return BreastDataAdapterUtils.apply_rectangular_spatial_cropping(df, cropping_details)
        else:
            return BreastDataAdapterUtils.apply_circular_spatial_cropping(df, cropping_details)
            
    @staticmethod
    def apply_rectangular_spatial_cropping(df: pd.DataFrame, cropping_details: CroppingDetails) -> pd.DataFrame:
        """
        Apply rectangular spatial cropping to the dataframe.
        
        Args:
            df: The DataFrame to filter.
            cropping_details: CroppingDetails object with rectangular crop configuration.
            
        Returns:
            Filtered DataFrame.
        """
        # Validate WSI IDs exist in dataframe
        for wsi_id in cropping_details.wsi_to_crops.keys():
            if wsi_id not in df['id'].unique():
                raise ValueError(f"WSI ID '{wsi_id}' not found in data.")
        
        # Apply spatial filtering
        filtered_rows = []
        for idx, row in df.iterrows():
            wsi_id = row['id']
            x, y = row['x_pixel'], row['y_pixel']
            rects = cropping_details.wsi_to_crops.get(wsi_id, set())
            
            # Check if point is inside any rectangle for this WSI
            inside = False
            for rect in rects:
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    inside = True
                    break
            
            # Keep or exclude based on method
            if (cropping_details.crop_method == 'outside' and not inside) or \
               (cropping_details.crop_method == 'inside' and inside):
                filtered_rows.append(idx) #type: ignore
        
        # Filter the dataframe
        return df.loc[filtered_rows].reset_index(drop=True)
    
    @staticmethod
    def apply_circular_spatial_cropping(df: pd.DataFrame, cropping_details: CroppingDetails) -> pd.DataFrame:
        """
        Apply circular spatial cropping to the dataframe.
        
        Args:
            df: The DataFrame to filter.
            cropping_details: CroppingDetails object with circular crop configuration.
            
        Returns:
            Filtered DataFrame.
        """
        # Validate WSI IDs exist in dataframe
        for wsi_id in cropping_details.wsi_to_crops.keys():
            if wsi_id not in df['id'].unique():
                raise ValueError(f"WSI ID '{wsi_id}' not found in data.")
        
        # Apply spatial filtering
        filtered_rows = []
        for idx, row in df.iterrows():
            wsi_id = row['id']
            x, y = row['x_pixel'], row['y_pixel']
            circles = cropping_details.wsi_to_crops.get(wsi_id, set())
            
            # Check if point is inside any circle for this WSI
            inside = False
            for circle in circles:
                center_x, center_y, radius = circle
                # Calculate Euclidean distance
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if distance <= radius:  # Points on boundary are considered inside
                    inside = True
                    break
            
            # Keep or exclude based on method
            if (cropping_details.crop_method == 'outside' and not inside) or \
               (cropping_details.crop_method == 'inside' and inside):
                filtered_rows.append(idx) #type: ignore
        
        # Filter the dataframe
        return df.loc[filtered_rows].reset_index(drop=True)

class BreastTrainingDataAdapter(BaseTrainingDataAdapter):
    """
    Data adapter for breast cancer spatial transcriptomics training data.
    
    This adapter loads data from a CSV file containing spatial coordinates,
    barcode identifiers, whole slide image IDs, and gene expression values.
    Optionally supports spatial filtering by rectangular crops.
    
    Attributes:
        name (str): Identifier for this adapter type.
        image_dir (str): Directory containing image patches.
        breast_csv (str): Path to the CSV file containing the data.
        df (pd.DataFrame): DataFrame containing the loaded data.
        wsi_ids (List[str]): List of whole slide image IDs to include.
        gene_ids (List[str]): List of gene IDs to include.
        gene_cols (List[str]): List of columns in the DataFrame corresponding to genes.
        cropping_details (Optional[CroppingDetails]): Spatial cropping configuration.
        aug_seq_list (Optional[List[List[BaseAugmentation]]]): List of augmentation sequences.
    """
    
    name = 'Breast Training Data Adapter'

    def __init__(self, image_dir: str, breast_csv: str, wsi_ids: list[str], gene_ids: list[str], 
                 cropping_details: Optional[CroppingDetails] = None,
                 aug_seq_list: Optional[List[List[BaseAugmentation]]] = None) -> None:
        """
        Initialize the breast training data adapter.
        
        Args:
            image_dir: Directory containing image patches.
            breast_csv: Path to the CSV file containing gene expression data.
            wsi_ids: List of whole slide image IDs to include.
            gene_ids: List of gene IDs to include.
            cropping_details: Optional CroppingDetails object for spatial filtering.
            aug_seq_list: Optional list of augmentation sequences to apply.
            
        Raises:
            ValueError: If the image directory doesn't exist, required WSI IDs are missing,
                        or duplicate gene IDs are provided.
            FileNotFoundError: If the CSV file doesn't exist or required columns are missing.
        """
        # Validate and initialize basic parameters
        BreastDataAdapterUtils.validate_image_dir(image_dir)
        BreastDataAdapterUtils.validate_csv_file(breast_csv)
        self.image_dir = image_dir
        self.breast_csv = breast_csv
        
        # Load DataFrame
        self.df: pd.DataFrame = pd.read_csv(self.breast_csv) #type: ignore
        
        # Validate and set WSI IDs
        self.wsi_ids = wsi_ids
        BreastDataAdapterUtils.validate_wsi_ids(self.wsi_ids, self.df)
        
        # Validate and set gene IDs
        self.gene_ids = sorted(gene_ids)
        self._validate_gene_ids()
        
        # Filter and prepare DataFrame
        self._prepare_dataframe()
        
        # Process spatial cropping if cropping_details is provided
        self.cropping_details = cropping_details
        
        if cropping_details:
            # Apply spatial cropping
            self.df = BreastDataAdapterUtils.apply_spatial_cropping(
                self.df, self.cropping_details
            )
        if aug_seq_list:
            self.aug_seq_list = aug_seq_list
            self._prepare_augmented_dataframe()
        else:
            self.df["augmentation"] = [None] * len(self.df)

    def _prepare_augmented_dataframe(self) -> None:
        """Prepare the augmented DataFrame by applying specified augmentations."""
        # Validation
        for aug_seq in self.aug_seq_list:
            for aug in aug_seq:
                if not isinstance(aug, BaseAugmentation): #type: ignore
                    raise TypeError(f"Expected BaseAugmentation instance, got {type(aug)}")

        # Base dataframe: original rows with augmentation=None
        base_df = self.df.copy(deep=True)
        base_df['augmentation'] = None

        # List to store all variations
        df_list = [base_df]

        # Add augmented copies
        for aug_seq in self.aug_seq_list:
            temp_df = base_df.copy(deep=True)
            # Create a list of the same augmentation sequence with length equal to DataFrame
            aug_seq_list_repeated = [aug_seq] * len(temp_df)
            temp_df['augmentation'] = aug_seq_list_repeated  # assign the repeated list to match DataFrame length
            df_list.append(temp_df)

        # Combine all into self.df
        self.df = pd.concat(df_list, ignore_index=True)

    def _validate_gene_ids(self) -> None:
        """Validate that gene IDs are unique and present in the dataset."""
        # Check for duplicates
        if len(self.gene_ids) != len(set(self.gene_ids)):
            duplicates = [gene for gene in self.gene_ids if self.gene_ids.count(gene) > 1]
            raise ValueError(ErrorMessage.duplicate_gene_ids(duplicates))
        
        # Filter dataframe first to avoid checking for columns in rows we won't use
        self.df = self.df[self.df['id'].isin(self.wsi_ids)] #type: ignore
        
        # Check for missing columns
        missing = [col for col in self.gene_ids if col not in self.df.columns]
        if missing:
            raise ValueError(ErrorMessage.missing_columns_in_dataframe(missing))
            
    def _prepare_dataframe(self) -> None:
        """Prepare the DataFrame by selecting relevant columns."""
        metadata_cols = ['barcode', 'id', 'x_pixel', 'y_pixel', 'combined_text']
        self.df = self.df[metadata_cols + self.gene_ids]
        self.gene_cols = [col for col in self.df.columns if col not in metadata_cols]

    def __getitem__(self, idx: int) -> TrainingDataPoint:
        """
        Get a training data point by index.
        
        Args:
            idx: Index of the data point to retrieve.
            
        Returns:
            A TrainingDataPoint instance containing the requested data.
            
        Raises:
            FileNotFoundError: If the image patch file doesn't exist.
        """
        row = self.df.iloc[idx]
        barcode = row['barcode']
        wsi_id = row['id']
        img_patch_path = BreastDataAdapterUtils.construct_image_patch_path(
            self.image_dir, barcode, wsi_id
        )
        
        # Create gene expression dictionary
        gene_expression = {gene: float(row[gene]) for gene in self.gene_cols}

        return TrainingDataPoint(
            x=row['x_pixel'],
            y=row['y_pixel'],
            img_patch_path=img_patch_path,
            gene_expression=gene_expression,
            wsi_id=wsi_id,
            barcode=barcode,
            aug_seq= row["augmentation"],
        )

    def __len__(self) -> int:
        """
        Get the number of data points available.
        
        Returns:
            Number of rows in the filtered DataFrame.
        """
        return len(self.df)
        
    def get_gene_ids(self) -> list[str]:
        """
        Get the list of gene IDs used by this adapter.
        
        Returns:
            List of gene IDs.
        """
        return self.gene_ids

class BreastPredictionDataAdapter(BasePredictionDataAdapter):
    """
    Data adapter for breast cancer spatial transcriptomics prediction data.
    
    This adapter loads data from a CSV file containing spatial coordinates,
    barcode identifiers, and whole slide image IDs, without gene expression values.
    Optionally supports spatial filtering by rectangular crops.
    
    Attributes:
        name (str): Identifier for this adapter type.
        image_dir (str): Directory containing image patches.
        prediction_csv (str): Path to the CSV file containing the data.
        df (pd.DataFrame): DataFrame containing the loaded data.
        wsi_ids (List[str]): List of whole slide image IDs to include.
        cropping_details (Optional[CroppingDetails]): Spatial cropping configuration.
    """
    
    name = 'Breast Prediction Data Adapter'
    
    def __init__(self, image_dir: str, prediction_csv: str, wsi_ids: list[str],
                 cropping_details: Optional[CroppingDetails] = None) -> None:
        """
        Initialize the breast prediction data adapter.
        
        Args:
            image_dir: Directory containing image patches.
            prediction_csv: Path to the CSV file containing prediction metadata.
            wsi_ids: List of whole slide image IDs to include.
            cropping_details: Optional CroppingDetails object for spatial filtering.
            
        Raises:
            ValueError: If the image directory doesn't exist or required WSI IDs are missing.
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If required columns are missing in the DataFrame.
        """
        # Validate and initialize basic parameters
        BreastDataAdapterUtils.validate_image_dir(image_dir)
        BreastDataAdapterUtils.validate_csv_file(prediction_csv)
        self.image_dir = image_dir
        self.prediction_csv = prediction_csv
        
        # Load DataFrame
        self.df: pd.DataFrame = pd.read_csv(self.prediction_csv) #type: ignore
        
        # Validate required columns
        self._validate_required_columns()
        
        # Validate and set WSI IDs
        self.wsi_ids = wsi_ids
        BreastDataAdapterUtils.validate_wsi_ids(self.wsi_ids, self.df)
        
        # Filter DataFrame
        self._prepare_dataframe()
        
        # Process spatial cropping if cropping_details is provided
        self.cropping_details = cropping_details
        
        if cropping_details:
            # Apply spatial cropping
            self.df = BreastDataAdapterUtils.apply_spatial_cropping(
                self.df, self.cropping_details
            )
            
    def _validate_required_columns(self) -> None:
        """Validate that the DataFrame has all required columns."""
        required_cols = ['barcode', 'id', 'x_pixel', 'y_pixel']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(ErrorMessage.missing_columns_in_dataframe(missing))
            
    def _prepare_dataframe(self) -> None:
        """Prepare the DataFrame by filtering to include only specified WSI IDs."""
        self.df = self.df[self.df['id'].isin(self.wsi_ids)] #type: ignore

    def __getitem__(self, idx: int) -> PredictionDataPoint:
        """
        Get a prediction data point by index.
        
        Args:
            idx: Index of the prediction data point to retrieve.
            
        Returns:
            A PredictionDataPoint instance containing the requested data.
            
        Raises:
            FileNotFoundError: If the image patch file doesn't exist.
        """
        row = self.df.iloc[idx]
        barcode = row['barcode']
        wsi_id = row['id']
        img_patch_path = BreastDataAdapterUtils.construct_image_patch_path(
            self.image_dir, barcode, wsi_id
        )
        
        return PredictionDataPoint(
            x=row['x_pixel'],
            y=row['y_pixel'],
            img_patch_path=img_patch_path,
            wsi_id=wsi_id,
            barcode=barcode,
        )

    def __len__(self) -> int:
        """
        Get the number of prediction data points available.
        
        Returns:
            Number of rows in the filtered DataFrame.
        """
        return len(self.df)