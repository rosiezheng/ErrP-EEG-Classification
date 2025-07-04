# ErrP-EEG-Classification
Classification of Error-related Potential (ErrP) from EEG signals

### Project Objective

In this mini project, we develop an offline brain–computer interface (BCI) classifier to detect Error-related Potentials (ErrPs) from EEG recordings. Leveraging a publicly available dataset, the aim is to explore flexible pipelines for data preprocessing, feature extraction, model design, and to present interpretable classification results.


### Dataset

We use the ErrP dataset shared by Chavarriaga and Millán (2010) [1], accessible under “22. Monitoring error-related potentials (013-2015)” at the BNCI Horizon 2020 portal:

https://bnci-horizon-2020.eu/database/data-sets

This dataset consists of EEG recordings from six participants, each undergoing two sessions separated by several weeks. During the experiment, subjects monitored an external agent’s decisions (correct or incorrect), eliciting ErrPs on error trials. 


### Implementation Details

This repository provides a modular pipeline covering:

1. Data Preprocessing & Aggregation (data_preprocessing_aggregation.ipynb) --> Bandpass filtering, baseline correction, epoching, and trial aggregation.
2. Feature Extraction & Selection (feature_extraction_selection.ipynb) --> Spatial filtering with Fisher Criterion Beamformer, for which we used the FCB Toolbox https://github.com/gpiresML/FCB-spatial-filter and [2]
3. Classification with extracted features and sLDA (classification_sLDA_FCB.ipynb) --> Simple model: shrinkage-regularized linear discriminant analysis.
4. Transformer Classification with GAN-augmented data (classification_GAN_Transformer.ipynb) --> Complex model: data augmentation via generative adversarial networks and classification with a transformer network.

### Utility Functions (bci_utils.py)

Comprehensive toolkit with functions including:

- EEG spatial filtering (Fisher Criterion Beamformer, FCB [2])
- r^2 feature evaluation and other feature selection tools
- Data augmentation with generative adversarial networks (GANs)
- Neural network models for EEG (Transformer, GAN Generator/Discriminator)
- Cross-validation methods (Stratified K-Fold, Leave-One-Subject-Out, Leave-One-Session-Out)
- Utility functions for results saving and serialization

### Usage

Clone the repository and navigate into its root directory:

git clone https://github.com/yourusername/ErrP-EEG-Classification.git
cd ErrP-EEG-Classification

Open the notebooks in JupyterLab or Jupyter Notebook.

Follow the execution order: preprocessing → feature extraction → classification.

### Results

Main performance metrics are stored in the results/ folder. 

### References

[1] Chavarriaga, R., & Millán, J. d. R. (2010). Monitoring error-related potentials in BCI: Corrigendum and update. Journal of Neural Engineering.

[2] Gabriel Pires, Urbano Nunes and Miguel Castelo-Branco (2011), "Statistical Spatial Filtering for a P300-based BCI: Tests in able-bodied, and Patients with Cerebral Palsy and Amyotrophic Lateral Sclerosis", Journal of Neuroscience Methods, Elsevier, 2011, 195(2), Feb. 2011: doi:10.1016/j.jneumeth.2010.11.016 https://www.sciencedirect.com/science/article/pii/S0165027010006503?via%3Dihub
