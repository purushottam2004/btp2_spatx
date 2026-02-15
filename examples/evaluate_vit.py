"""
Script to evaluate the trained ViT model on validation set and get Pearson scores.
"""

import os
import logging
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import ViT components
from spatx_core.data.data import TrainingData
from spatx_core.datasets.vit_to_gene import ViTTrainingDataset
from spatx_core.models.vit_to_gene import create_vit_gene_model
from spatx_core.models.vit_to_gene._loss import CombinedLoss as ViTCombinedLoss
from spatx_core.data_adapters.hest_data_adapter import HestTrainingDataAdapter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gene IDs for lung dataset (same as training)
gene_ids = [
    "S100A2","PTGDS","CD86","EGFR","SCGB3A2","CCL18","BAX","UCHL3","XBP1",
    "RHOA","HLA-DRA","PTPRC","COL1A1","SNAI1","FN1","HIF1A","VIM","GZMA",
    "LAMP3","RNASE1","CEACAM6","IL1B","SEC11C","CCN2","CDK1","SFTPC",
    "PDIA4","HSPA5","PDIA6","MS4A7","PLIN2","FGF2","FGFBP2","LYZ",
    "SPARCL1","ICAM1","ATF4","EPCAM","MGST1","DCN","ASCL1","SCGB1A1",
    "CD44","CD68","ATF3","HAS2","CD1A","BPIFA1","CD14","GCLM","PDIA3",
    "HERPUD1","EHMT1","UBE2J1","COL1A2","IDH1","ERLEC1","HYOU1","SPCS3",
    "SPCS2","COL3A1","HMGA1","CCL2","IRF7","IRF1","NUCB2","ITGAV",
    "S100A12","COL4A3","HMOX1","AKR1C1","SSR3","KRT18","MCEMP1","KLRG1",
    "BCL2L1","EPAS1","KLRC1","FABP4","ITGB1","MAL","PRDX4","AXIN2",
    "CTNNB1","SPP1","VEGFA","PKM","HIST1H1C","KRT15","STAT1","ITGAM",
    "PCNA","FCER1G","KRT8","CCR7","MYC","ANKRD28","AGER","SFTPD",
    "NHSL2","MRC1","SOD2","ATF6","DNAJB9","ACTA2","FCN1","FAS",
    "PECAM1","AIF1","CCL21","CD34","CD52","PPARG","SMAD4","ATG7",
    "NAPSA","TGFB1","WFDC2","BMP4","SFTA2","NKX2-1","CD8A","PDGFRB",
    "HAVCR2","WWTR1","LUM","MARCO","AGR3","HES1","UQCRHL","SOX2",
    "DEFB1","PGC","LMAN1","TP73","WNT3A","RETN","CD27","YAP1","CST3",
    "KRT17","SLC25A4","LGALS1","SPRY2","SELENOS","MYDGF","POSTN",
    "DUOX1","S100A8","CRELD2","ITM2C","BANK1","NUTF2","ITGA3","PLPP5",
    "TOP1","ITGB6","STAT6","CHAC1","BMPR2","TGFB3","WNT7B","NFKB1",
    "HLA-DQB1","CEACAM5","FGF7","LPAR1","SCG2","RTKN2","MANF","GSR",
    "ZEB1","SOX4","C1QC","IL32","DDIT3","SFRP2","GZMB","SFRP4","PIM2",
    "S100A9","GKN2","TGFB2","DMBT1","SLC25A37","IL4R","IFIT2","PDGFRA",
    "KDR","MUC5B","TRAC","GDF15","HLA-DQA1","CD4","WNT2","IFIT3",
    "ISG20","CTHRC1","DCTPP1","GPR183","LTF","KIT","TTC19","GNG11",
    "FCGR3A","WNT5A","BCL2","MMP7","MSLN","IFIT1","COL15A1","IL7R",
    "TBXA2R","AXL","FOXI1","CD247","OAS2","GLP1R","CXCR4","CXCL9",
    "ITGAX","CLDN5","FAP","TPSAB1","UGDH","TNFRSF13C","FGF10","MEG3",
    "LEF1","CTLA4","KRT6A","CA4","FCER1A","KRT5","FOXJ1","CSPG4",
    "RAMP2","OAS3","HAS1","FCN3","BCL2L11","ATP2A3","KLRB1","CD69",
    "CD274","APLN","CDKN2A","ACKR1","CPA3","CD1C","JCHAIN","SNAI2",
    "AKR1C2","NOX4","TREM2","IFNG","SNCA","SPINK1","CCNB2","RSPO3",
    "CFTR","CCNA1","SLC1A3","CD8B","AKR1B10","CDH26","LPAR2","LGR5",
    "EREG","FKBP11","SAA2","SLC2A1","RACGAP1","UBE2S","CALCA","CD2",
    "CD3G","MS4A1","KRT14","LAG3","GNLY","CHGB","PAEP","S100A7",
    "SOX9","IL1A","LTB","CREB3L4","IL37","ITGAE","ELN","MKI67",
    "PLVAP","HEY1","TP63","LGR6","CD79B","CD28","IL2RA","CCL5",
    "CD79A","CD3E","CXCL13","TNFRSF17","CD19","CCL22","TOP2A",
    "NKG7","WT1","TNFRSF9","CXCL14","APLNR","TERT","CENPF","MMP12",
    "CXCR5","IL11","CD3D","GZMK","SLC7A11","MMP10","LCK","ABCC2",
    "IGLL1","MUC5AC","LILRA4","ELANE","DIRAS3","PI16","FASLG",
    "ERN2","MFAP5","PDCD1","TCL1A","CGA","TNF","LY6D","FOXP3",
    "VPREB3"
]

lung_wsi_ids = [
    "NCBI856", "NCBI858", "NCBI860", "NCBI866", "NCBI870",
    "NCBI875", "NCBI879", "NCBI881", "NCBI883", "NCBI857",
    "NCBI859", "NCBI861", "NCBI867", "NCBI873", "NCBI876",
    "NCBI880", "NCBI882", "NCBI884"
]


def calculate_pearson_scores(predictions: np.ndarray, targets: np.ndarray) -> tuple:
    """
    Calculate per-gene Pearson correlation scores.
    """
    num_genes = predictions.shape[1]
    per_gene_pearson = []
    
    for g in range(num_genes):
        try:
            result = pearsonr(predictions[:, g].tolist(), targets[:, g].tolist())
            pc = result.correlation
            per_gene_pearson.append(float(pc) if not np.isnan(pc) else 0.0)
        except Exception as e:
            logger.debug(f"Failed to calculate Pearson correlation for gene {g}: {e}")
            per_gene_pearson.append(0.0)
    
    mean_pearson = float(np.mean(per_gene_pearson))
    return per_gene_pearson, mean_pearson


def evaluate_model():
    """
    Load the best model and evaluate on validation set.
    """
    # Configuration
    hest_data_dir = Path("/mnt/wwn-0x5000c500e655a860/purushottam/btp2_spatx/hest_data_new")
    model_path = "saved_models/vit_to_gene/experiment_1/best_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    num_workers = 4
    num_genes = len(gene_ids)
    
    # Check model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    logger.info("=" * 60)
    logger.info("ViT Model Evaluation on Validation Set")
    logger.info("=" * 60)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Number of genes: {num_genes}")
    
    # Create validation data adapter
    validation_adapter = HestTrainingDataAdapter(
        base_dir=str(hest_data_dir),
        wsi_ids=lung_wsi_ids,
        gene_ids=gene_ids,
    )
    
    if len(validation_adapter) == 0:
        raise ValueError("Empty validation dataset")
    
    # Create dataset and dataloader
    validation_data = TrainingData(validation_adapter)
    validation_dataset = ViTTrainingDataset(validation_data)
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"Validation dataset: {len(validation_dataset)} samples")
    logger.info("=" * 60)
    
    # Load model
    model = create_vit_gene_model(
        model_size='base',
        num_genes=num_genes,
        head_type='simple',
        pretrained=False
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Initialize loss function
    criterion = ViTCombinedLoss(alpha=0.5, beta=1.0, reg_lambda=0.0).to(device)
    
    # Evaluate
    val_loss = 0.0
    val_predictions = []
    val_targets = []
    
    logger.info("Running evaluation...")
    with torch.no_grad():
        for images, expressions, _ in tqdm(validation_dataloader, desc="Evaluating"):
            images = images.to(device)
            expressions = expressions.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, expressions)
            val_loss += loss.item()
            
            val_predictions.append(predictions.cpu())
            val_targets.append(expressions.cpu())
    
    avg_val_loss = val_loss / len(validation_dataloader)
    
    # Calculate Pearson scores
    val_preds_np = torch.cat(val_predictions, dim=0).numpy()
    val_targets_np = torch.cat(val_targets, dim=0).numpy()
    val_per_gene_pearson, val_mean_pearson = calculate_pearson_scores(val_preds_np, val_targets_np)
    
    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Validation Loss: {avg_val_loss:.6f}")
    logger.info(f"Mean Pearson Correlation: {val_mean_pearson:.6f}")
    logger.info("=" * 60)
    logger.info("\nPer-Gene Pearson Correlations:")
    logger.info("-" * 40)
    
    # Sort genes by Pearson score for better readability
    gene_scores = list(zip(gene_ids, val_per_gene_pearson))
    gene_scores_sorted = sorted(gene_scores, key=lambda x: x[1], reverse=True)
    
    for gene_id, score in gene_scores_sorted:
        logger.info(f"  {gene_id:15s}: {score:.6f}")
    
    logger.info("=" * 60)
    
    # Summary statistics
    pearson_array = np.array(val_per_gene_pearson)
    logger.info("\nSummary Statistics:")
    logger.info(f"  Mean:   {np.mean(pearson_array):.6f}")
    logger.info(f"  Std:    {np.std(pearson_array):.6f}")
    logger.info(f"  Median: {np.median(pearson_array):.6f}")
    logger.info(f"  Min:    {np.min(pearson_array):.6f}")
    logger.info(f"  Max:    {np.max(pearson_array):.6f}")
    logger.info(f"  Genes with r > 0.3: {np.sum(pearson_array > 0.3)}/{len(pearson_array)}")
    logger.info(f"  Genes with r > 0.5: {np.sum(pearson_array > 0.5)}/{len(pearson_array)}")
    logger.info("=" * 60)
    
    return avg_val_loss, val_mean_pearson, val_per_gene_pearson


if __name__ == "__main__":
    print("ViT Model Evaluation Script")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    evaluate_model()
    
    print("\nDone!")
