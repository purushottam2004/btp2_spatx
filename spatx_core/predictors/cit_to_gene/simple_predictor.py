import os
import logging
import importlib.util
import csv
from typing import List

import torch
from torch.utils.data import DataLoader

from ...data.data import PredictionData
from ...data_adapters import BreastPredictionDataAdapter
from ...datasets.cit_to_gene import CITPredictionDataset
from ...models.cit_to_gene.cit import CIT
from ...models.cit_to_gene.CiTGene import CITGenePredictor

logger = logging.getLogger(__name__)

class PredictionResults:
    def __init__(self, gene_ids: list[str], results_dir: str | None = None):
        self.gene_ids = gene_ids
        self.predictions = []  # List of dicts with prediction data
        self.results_dir = None
        self.save_path = None
        
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            self.results_dir = results_dir
            if not os.access(os.path.dirname(results_dir), os.W_OK):
                raise ValueError(f"Results path {results_dir} is not writable")
            self.save_path = os.path.join(self.results_dir, "predictions.csv")

    def add_prediction(self, sample_info: dict, gene_predictions: torch.Tensor):
        """Add a prediction result"""
        prediction_dict = {
            'image_name': sample_info['image_name'],
            'x': sample_info['x'],
            'y': sample_info['y'],
        }
        
        # Add gene predictions
        gene_preds = gene_predictions.cpu().numpy()
        for i, gene_id in enumerate(self.gene_ids):
            prediction_dict[gene_id] = float(gene_preds[i])
            
        self.predictions.append(prediction_dict)
    
    def save_predictions(self):
        """Save predictions to CSV file"""
        if not self.save_path or not self.predictions:
            return
            
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Define CSV columns
        columns = ['image_name', 'x', 'y'] + self.gene_ids
        
        # Write to CSV
        with open(self.save_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.predictions)
            
        logger.info(f"Predictions saved to: {self.save_path}")

class SimpleCITPredictor:
    def __init__(self, 
                 prediction_adapter: BreastPredictionDataAdapter,
                 model_path : str,
                 required_gene_ids: List[str],
                 model_genes_path: str,
                 device: str = 'cuda',
                 results_dir: str | None = None,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 logging_interval : int = 10
                 ):
        if len(prediction_adapter) == 0:
            raise ValueError("Empty prediction dataset detected")
        if batch_size > len(prediction_adapter):
            raise ValueError("Batch size cannot be larger than prediction dataset size")
        if device not in ['cpu', 'cuda']:
            raise ValueError(f"The device must be cpu or cuda, currently it is {device}")
        if device == 'cuda' and (not torch.cuda.is_available()):
            raise ValueError("Cuda is not available, please don't use")

        self.model_path = model_path
        self.gene_ids_file = model_genes_path

        if not os.path.exists(self.model_path):
            raise ValueError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.gene_ids_file):
            raise ValueError(f"Gene IDs file not found at {self.gene_ids_file}")
        
        # Load and validate gene IDs
        self.gene_ids = self._load_gene_ids()
        self._validate_gene_ids(required_gene_ids)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.results_dir = results_dir if results_dir else None
        self.results = PredictionResults(gene_ids=self.gene_ids, results_dir=self.results_dir)
        self.prediction_data = PredictionData(prediction_adapter)
        self.num_genes = len(self.gene_ids)
        self.logging_interval = logging_interval

    def _load_gene_ids(self) -> list[str]:
        """Load gene IDs from the model's gene IDs file"""
        spec = importlib.util.spec_from_file_location("gene_ids_module", self.gene_ids_file)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load gene IDs from {self.gene_ids_file}")
        
        gene_ids_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gene_ids_module)
        
        if not hasattr(gene_ids_module, 'gene_ids'):
            raise ValueError(f"No 'gene_ids' variable found in {self.gene_ids_file}")
        
        gene_ids = gene_ids_module.gene_ids
        if not isinstance(gene_ids, list):
            raise ValueError(f"gene_ids must be a list, found {type(gene_ids)}")
        
        return gene_ids

    def _validate_gene_ids(self, required_gene_ids: list[str]):
        """Validate that required gene IDs are present in the model's gene IDs"""
        model_gene_set = set(self.gene_ids)
        required_gene_set = set(required_gene_ids)
        
        missing_genes = required_gene_set - model_gene_set
        if missing_genes:
            raise ValueError(f"The following required gene IDs are not present in the model: {sorted(missing_genes)}")

    def predict(self):
        # Initialize dataset and dataloader
        prediction_dataset = CITPredictionDataset(self.prediction_data)
        prediction_dataloader = DataLoader(dataset=prediction_dataset, batch_size=self.batch_size, 
                                         shuffle=False, num_workers=self.num_workers)
        
        # Log prediction initialization
        logger.info("=" * 60)
        logger.info("Prediction Initialization")
        logger.info("=" * 60)
        logger.info(f"Starting prediction")
        logger.info(f"Prediction dataset: {len(prediction_dataset)} samples")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model Path: {self.model_path}")
        logger.info(f"Number of genes: {self.num_genes}")
        logger.info(f"Model path: {self.model_path}")
        if self.results.save_path:
            logger.info(f"Results will be saved to: {self.results.save_path}")
        else:
            logger.info("Results will not be saved to file")
        logger.info("=" * 60)
        logger.info(f"Log interval: {self.logging_interval} batch(es)")

        
        # Initialize model and load state dict
        cit_backbone = CIT(img_size=224, in_chans=3,
                          embed_dim=96,
                          depths=[2,2,6,2],
                          num_heads=[3,6,12,24],
                          device=self.device)
        model = CITGenePredictor(cit_backbone, num_genes=self.num_genes)
        
        # Load model state dict
        logger.info("Loading model from state dict...")
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        logger.info("Model loaded successfully")
        
        # Prediction phase
        logger.info("Starting prediction phase")
        
        with torch.no_grad():
            for batch_idx, (image_tensor, image_name, x, y, _, _) in enumerate(prediction_dataloader):
                # Move batch to device
                image_tensor = image_tensor.to(self.device)

                # Forward pass
                predictions = model(image_tensor)
                
                # Process each sample in the batch
                for i in range(image_tensor.shape[0]):  # Use batch size instead of len(sample_infos['image_name'])
                    sample_info = {
                        'image_name': image_name[i],
                        'x': x[i].item(),
                        'y': y[i].item(),
                    }
                    gene_predictions = predictions[i]
                    self.results.add_prediction(sample_info, gene_predictions)
                
                # Log progress
                if (batch_idx + 1) % self.logging_interval == 0:
                    progress = (batch_idx + 1) / len(prediction_dataloader) * 100
                    logger.info(f"Batch {batch_idx + 1}/{len(prediction_dataloader)} processed ({progress:.1f}%)")
        
        logger.info(f"Prediction phase completed")
        
        # Save predictions
        if self.results.save_path:
            self.results.save_predictions()
        
        # Prediction completion logging
        logger.info("=" * 60)
        logger.info("Prediction Completed Successfully")
        logger.info("=" * 60)
        logger.info(f"Total predictions made: {len(self.results.predictions)}")
        logger.info(f"Genes predicted: {len(self.gene_ids)}")
        if self.results.save_path:
            logger.info(f"Results saved to: {self.results.save_path}")
        logger.info("=" * 60)
        
        return self.results
