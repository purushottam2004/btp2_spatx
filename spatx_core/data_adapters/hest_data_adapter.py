"""
Module providing data adapters for distributed HEST spatial transcriptomics data.

This module implements adapters for data organized in a distributed folder structure
where each WSI has its own subfolder containing patch data CSV and images.

Folder structure expected:
    base_dir/
    ├── WSI_ID_1/
    │   ├── WSI_ID_1_patch_data.csv
    │   └── 20x/
    │       ├── barcode1_WSI_ID_1.png
    │       └── ...
    ├── WSI_ID_2/
    │   ├── WSI_ID_2_patch_data.csv
    │   └── 20x/
    │       └── ...
    └── ...
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


class HestDataAdapterUtils:
    """
    Utility class containing common functionality for HEST data adapters.
    """
    
    @staticmethod
    def validate_base_dir(base_dir: str) -> None:
        """Validate that the base directory exists."""
        if not os.path.exists(base_dir):
            raise ValueError(ErrorMessage.directory_not_exists(base_dir))
        if not os.path.isdir(base_dir):
            raise ValueError(f"Path {base_dir} is not a directory")
    
    @staticmethod
    def get_wsi_folder_path(base_dir: str, wsi_id: str) -> str:
        """Get the folder path for a specific WSI."""
        return os.path.join(base_dir, wsi_id)
    
    @staticmethod
    def get_csv_path(base_dir: str, wsi_id: str) -> str:
        """Get the CSV file path for a specific WSI."""
        return os.path.join(base_dir, wsi_id, f"{wsi_id}_patch_data.csv")
    
    @staticmethod
    def get_image_dir(base_dir: str, wsi_id: str) -> str:
        """Get the image directory path for a specific WSI."""
        return os.path.join(base_dir, wsi_id, "20x")
    
    @staticmethod
    def validate_wsi_folder(base_dir: str, wsi_id: str) -> None:
        """
        Validate that a WSI folder exists and contains required files.
        
        Args:
            base_dir: Base directory containing all WSI folders.
            wsi_id: The WSI ID to validate.
            
        Raises:
            ValueError: If the folder doesn't exist.
            FileNotFoundError: If required files are missing.
        """
        wsi_folder = HestDataAdapterUtils.get_wsi_folder_path(base_dir, wsi_id)
        if not os.path.exists(wsi_folder):
            raise ValueError(f"WSI folder not found: {wsi_folder}")
        
        csv_path = HestDataAdapterUtils.get_csv_path(base_dir, wsi_id)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        image_dir = HestDataAdapterUtils.get_image_dir(base_dir, wsi_id)
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
    
    @staticmethod
    def discover_wsi_ids(base_dir: str) -> List[str]:
        """
        Discover all WSI IDs by scanning subdirectories.
        
        Args:
            base_dir: Base directory containing WSI folders.
            
        Returns:
            List of discovered WSI IDs.
        """
        wsi_ids = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Check if this looks like a valid WSI folder
                csv_path = HestDataAdapterUtils.get_csv_path(base_dir, item)
                image_dir = HestDataAdapterUtils.get_image_dir(base_dir, item)
                if os.path.exists(csv_path) and os.path.exists(image_dir):
                    wsi_ids.append(item)
        return sorted(wsi_ids)
    
    @staticmethod
    def construct_image_patch_path(base_dir: str, barcode: str, wsi_id: str) -> str:
        """
        Construct the image patch path for a given barcode and WSI ID.
        
        Args:
            base_dir: Base directory containing WSI folders.
            barcode: The barcode identifier.
            wsi_id: The whole slide image ID.
            
        Returns:
            The full path to the image patch.
            
        Raises:
            FileNotFoundError: If the image patch file doesn't exist.
        """
        image_dir = HestDataAdapterUtils.get_image_dir(base_dir, wsi_id)
        img_patch_name = f"{barcode}_{wsi_id}.png"
        img_patch_path = os.path.join(image_dir, img_patch_name)
        if not os.path.exists(img_patch_path):
            raise FileNotFoundError(ErrorMessage.image_patch_not_found(img_patch_path))
        return img_patch_path
    
    @staticmethod
    def load_combined_dataframe(base_dir: str, wsi_ids: List[str]) -> pd.DataFrame:
        """
        Load and combine DataFrames from multiple WSI folders.
        
        Args:
            base_dir: Base directory containing WSI folders.
            wsi_ids: List of WSI IDs to load.
            
        Returns:
            Combined DataFrame with all data.
        """
        dfs = []
        for wsi_id in wsi_ids:
            csv_path = HestDataAdapterUtils.get_csv_path(base_dir, wsi_id)
            df = pd.read_csv(csv_path)
            # Ensure WSI ID is set (in case it's not in the CSV)
            if 'id' not in df.columns:
                df['id'] = wsi_id
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
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
            return HestDataAdapterUtils.apply_rectangular_spatial_cropping(df, cropping_details)
        else:
            return HestDataAdapterUtils.apply_circular_spatial_cropping(df, cropping_details)
            
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
        filtered_rows = []
        for idx, row in df.iterrows():
            wsi_id = row['id']
            x, y = row['x_pixel'], row['y_pixel']
            rects = cropping_details.wsi_to_crops.get(wsi_id, set())
            
            inside = False
            for rect in rects:
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    inside = True
                    break
            
            if (cropping_details.crop_method == 'outside' and not inside) or \
               (cropping_details.crop_method == 'inside' and inside):
                filtered_rows.append(idx)  # type: ignore
        
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
        filtered_rows = []
        for idx, row in df.iterrows():
            wsi_id = row['id']
            x, y = row['x_pixel'], row['y_pixel']
            circles = cropping_details.wsi_to_crops.get(wsi_id, set())
            
            inside = False
            for circle in circles:
                center_x, center_y, radius = circle
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if distance <= radius:
                    inside = True
                    break
            
            if (cropping_details.crop_method == 'outside' and not inside) or \
               (cropping_details.crop_method == 'inside' and inside):
                filtered_rows.append(idx)  # type: ignore
        
        return df.loc[filtered_rows].reset_index(drop=True)


class HestTrainingDataAdapter(BaseTrainingDataAdapter):
    """
    Data adapter for distributed HEST spatial transcriptomics training data.
    
    This adapter loads data from multiple WSI folders, each containing their own
    CSV file and image directory. Data is combined into a single DataFrame for training.
    
    Expected folder structure:
        base_dir/
        ├── WSI_ID_1/
        │   ├── WSI_ID_1_patch_data.csv
        │   └── 20x/
        ├── WSI_ID_2/
        │   ├── WSI_ID_2_patch_data.csv
        │   └── 20x/
        └── ...
    
    Attributes:
        name (str): Identifier for this adapter type.
        base_dir (str): Base directory containing all WSI folders.
        df (pd.DataFrame): Combined DataFrame from all WSI folders.
        wsi_ids (List[str]): List of WSI IDs to include.
        gene_ids (List[str]): List of gene IDs to include.
        gene_cols (List[str]): List of columns corresponding to genes.
        cropping_details (Optional[CroppingDetails]): Spatial cropping configuration.
        aug_seq_list (Optional[List[List[BaseAugmentation]]]): List of augmentation sequences.
    """
    
    name = 'HEST Training Data Adapter'

    def __init__(self, base_dir: str, wsi_ids: Optional[List[str]], gene_ids: List[str], 
                 cropping_details: Optional[CroppingDetails] = None,
                 aug_seq_list: Optional[List[List[BaseAugmentation]]] = None) -> None:
        """
        Initialize the HEST training data adapter.
        
        Args:
            base_dir: Base directory containing all WSI folders.
            wsi_ids: List of WSI IDs to include. If None, auto-discovers all valid WSI folders.
            gene_ids: List of gene IDs to include.
            cropping_details: Optional CroppingDetails object for spatial filtering.
            aug_seq_list: Optional list of augmentation sequences to apply.
            
        Raises:
            ValueError: If the base directory doesn't exist, required WSI folders are missing,
                        or duplicate gene IDs are provided.
            FileNotFoundError: If required files don't exist.
        """
        # Validate base directory
        HestDataAdapterUtils.validate_base_dir(base_dir)
        self.base_dir = base_dir
        
        # Auto-discover or validate WSI IDs
        if wsi_ids is None:
            self.wsi_ids = HestDataAdapterUtils.discover_wsi_ids(base_dir)
            if not self.wsi_ids:
                raise ValueError(f"No valid WSI folders found in {base_dir}")
        else:
            self.wsi_ids = wsi_ids
            for wsi_id in self.wsi_ids:
                HestDataAdapterUtils.validate_wsi_folder(base_dir, wsi_id)
        
        # Load combined DataFrame
        self.df = HestDataAdapterUtils.load_combined_dataframe(base_dir, self.wsi_ids)
        
        if len(self.df) == 0:
            raise ValueError(f"No data found for WSI IDs: {self.wsi_ids}")
        
        # Validate and set gene IDs
        self.gene_ids = sorted(gene_ids)
        self._validate_gene_ids()
        
        # Prepare DataFrame
        self._prepare_dataframe()
        
        # Apply spatial cropping
        self.cropping_details = cropping_details
        if cropping_details:
            self.df = HestDataAdapterUtils.apply_spatial_cropping(self.df, cropping_details)
        
        # Apply augmentations
        if aug_seq_list:
            self.aug_seq_list = aug_seq_list
            self._prepare_augmented_dataframe()
        else:
            self.df["augmentation"] = [None] * len(self.df)

    def _prepare_augmented_dataframe(self) -> None:
        """Prepare the augmented DataFrame by applying specified augmentations."""
        for aug_seq in self.aug_seq_list:
            for aug in aug_seq:
                if not isinstance(aug, BaseAugmentation):  # type: ignore
                    raise TypeError(f"Expected BaseAugmentation instance, got {type(aug)}")

        base_df = self.df.copy(deep=True)
        base_df['augmentation'] = None
        df_list = [base_df]

        for aug_seq in self.aug_seq_list:
            temp_df = base_df.copy(deep=True)
            aug_seq_list_repeated = [aug_seq] * len(temp_df)
            temp_df['augmentation'] = aug_seq_list_repeated
            df_list.append(temp_df)

        self.df = pd.concat(df_list, ignore_index=True)

    def _validate_gene_ids(self) -> None:
        """Validate that gene IDs are unique and present in the dataset."""
        if len(self.gene_ids) != len(set(self.gene_ids)):
            duplicates = [gene for gene in self.gene_ids if self.gene_ids.count(gene) > 1]
            raise ValueError(ErrorMessage.duplicate_gene_ids(duplicates))
        
        missing = [col for col in self.gene_ids if col not in self.df.columns]
        if missing:
            raise ValueError(ErrorMessage.missing_columns_in_dataframe(missing))
            
    def _prepare_dataframe(self) -> None:
        """Prepare the DataFrame by selecting relevant columns."""
        # Check for optional combined_text column
        metadata_cols = ['barcode', 'id', 'x_pixel', 'y_pixel']
        if 'combined_text' in self.df.columns:
            metadata_cols.append('combined_text')
        
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
        img_patch_path = HestDataAdapterUtils.construct_image_patch_path(
            self.base_dir, barcode, wsi_id
        )
        
        gene_expression = {gene: float(row[gene]) for gene in self.gene_cols}

        return TrainingDataPoint(
            x=row['x_pixel'],
            y=row['y_pixel'],
            img_patch_path=img_patch_path,
            gene_expression=gene_expression,
            wsi_id=wsi_id,
            barcode=barcode,
            aug_seq=row["augmentation"],
        )

    def __len__(self) -> int:
        """
        Get the number of data points available.
        
        Returns:
            Number of rows in the filtered DataFrame.
        """
        return len(self.df)
        
    def get_gene_ids(self) -> List[str]:
        """
        Get the list of gene IDs used by this adapter.
        
        Returns:
            List of gene IDs.
        """
        return self.gene_ids


class HestPredictionDataAdapter(BasePredictionDataAdapter):
    """
    Data adapter for distributed HEST spatial transcriptomics prediction data.
    
    This adapter loads data from multiple WSI folders for making predictions,
    without requiring gene expression values.
    
    Expected folder structure:
        base_dir/
        ├── WSI_ID_1/
        │   ├── WSI_ID_1_patch_data.csv
        │   └── 20x/
        └── ...
    
    Attributes:
        name (str): Identifier for this adapter type.
        base_dir (str): Base directory containing all WSI folders.
        df (pd.DataFrame): Combined DataFrame from all WSI folders.
        wsi_ids (List[str]): List of WSI IDs to include.
        cropping_details (Optional[CroppingDetails]): Spatial cropping configuration.
    """
    
    name = 'HEST Prediction Data Adapter'
    
    def __init__(self, base_dir: str, wsi_ids: Optional[List[str]] = None,
                 cropping_details: Optional[CroppingDetails] = None) -> None:
        """
        Initialize the HEST prediction data adapter.
        
        Args:
            base_dir: Base directory containing all WSI folders.
            wsi_ids: List of WSI IDs to include. If None, auto-discovers all valid WSI folders.
            cropping_details: Optional CroppingDetails object for spatial filtering.
            
        Raises:
            ValueError: If the base directory doesn't exist or required WSI folders are missing.
            FileNotFoundError: If required files don't exist.
        """
        # Validate base directory
        HestDataAdapterUtils.validate_base_dir(base_dir)
        self.base_dir = base_dir
        
        # Auto-discover or validate WSI IDs
        if wsi_ids is None:
            self.wsi_ids = HestDataAdapterUtils.discover_wsi_ids(base_dir)
            if not self.wsi_ids:
                raise ValueError(f"No valid WSI folders found in {base_dir}")
        else:
            self.wsi_ids = wsi_ids
            for wsi_id in self.wsi_ids:
                HestDataAdapterUtils.validate_wsi_folder(base_dir, wsi_id)
        
        # Load combined DataFrame
        self.df = HestDataAdapterUtils.load_combined_dataframe(base_dir, self.wsi_ids)
        
        if len(self.df) == 0:
            raise ValueError(f"No data found for WSI IDs: {self.wsi_ids}")
        
        # Validate required columns
        self._validate_required_columns()
        
        # Prepare DataFrame
        self._prepare_dataframe()
        
        # Apply spatial cropping
        self.cropping_details = cropping_details
        if cropping_details:
            self.df = HestDataAdapterUtils.apply_spatial_cropping(self.df, cropping_details)
            
    def _validate_required_columns(self) -> None:
        """Validate that the DataFrame has all required columns."""
        required_cols = ['barcode', 'id', 'x_pixel', 'y_pixel']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(ErrorMessage.missing_columns_in_dataframe(missing))
            
    def _prepare_dataframe(self) -> None:
        """Prepare the DataFrame by selecting relevant columns."""
        metadata_cols = ['barcode', 'id', 'x_pixel', 'y_pixel']
        available_cols = [col for col in metadata_cols if col in self.df.columns]
        self.df = self.df[available_cols]

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
        img_patch_path = HestDataAdapterUtils.construct_image_patch_path(
            self.base_dir, barcode, wsi_id
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
