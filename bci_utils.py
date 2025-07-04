"""
BCI Utilities for EEG Error Potential Classification
---------------------------------------------------

Comprehensive toolkit for brain-computer interface (BCI) pipelines, including:

- EEG spatial filtering (Fisher Criterion Beamformer, FCB)
- r^2 feature evaluation and other feature selection tools
- Data augmentation with generative adversarial networks (GANs)
- Neural network models for EEG (Transformer, GAN Generator/Discriminator)
- Cross-validation methods (Stratified K-Fold, Leave-One-Subject-Out, Leave-One-Session-Out)
- Utility functions for results saving and serialization

Parts of this code are adapted from:
    - https://github.com/gpiresML/FCB-spatial-filter
      (Original author: Gabriel Pires, 2022; License: GNU GPL)

Cleaning, new code, and integration by Rosie Zheng, 2025.

If you use these functions in academic work, please cite the original repository and authors.

"""

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
import json

def FCB_spatial_filters(z1, z2, th):
    """
    Fisher Criterion Beamform (FCB) spatial filters.

    - The spatial BETWEEN-CLASS matrix (S_b) and WITHIN-CLASS matrix (S_w) are calculated in the spatial domain.
    - Spatial filters (eigenvectors) and their eigenvalues are obtained by solving the Rayleigh quotient:
        J(W) = (W' S_b W) / (W' S_w W)
        where W is the spatial filter.
    - The eigenvectors maximizing this criterion correspond to spatial filters that best discriminate between classes.

    Adapted from: https://github.com/gpiresML/FCB-spatial-filter
    Original author: Gabriel Pires, 2022

    Tidied by Rosie Zheng, 2025
    
    Args:
    z1, z2 : np.array
        EEG data arrays for class 1 and class 2, shape = (channels, samples, trials)
    th : float
        Regularization parameter for within-class covariance
    
    Returns:
    U1 : np.array
        Spatial filters (ordered eigenvectors)
    V1 : np.array
        Corresponding eigenvalues (diagonal matrix, ordered)
    """

    # Mean over trials per class
    Mean1 = np.mean(z1, axis=2)
    Mean2 = np.mean(z2, axis=2)

    n_chan = z1.shape[0]

    # Initialize covariance matrices
    Cov1 = np.zeros((n_chan, n_chan, z1.shape[2]))
    Cov2 = np.zeros((n_chan, n_chan, z2.shape[2]))

    # Compute normalized spatial covariance per trial, class 1
    for i in range(z1.shape[2]):
        aux = (z1[:, :, i] - Mean1) @ (z1[:, :, i] - Mean1).T
        Cov1[:, :, i] = aux / np.trace(aux)

    # Compute normalized spatial covariance per trial, class 2
    for i in range(z2.shape[2]):
        aux = (z2[:, :, i] - Mean2) @ (z2[:, :, i] - Mean2).T
        Cov2[:, :, i] = aux / np.trace(aux)

    # Prior probabilities for unbalanced classes
    n_trials_1 = z1.shape[2]
    n_trials_2 = z2.shape[2]
    p1 = n_trials_1 / (n_trials_1 + n_trials_2)
    p2 = n_trials_2 / (n_trials_1 + n_trials_2)

    # Average covariance over all trials per class
    Covavg1 = np.sum(Cov1, axis=2)
    Covavg2 = np.sum(Cov2, axis=2)

    # Weighted mean over both classes
    MeanAll = p1 * Mean1 + p2 * Mean2

    # Spatial BETWEEN-CLASS matrix
    Sb = (
        p1 * (Mean1 - MeanAll) @ (Mean1 - MeanAll).T
        + p2 * (Mean2 - MeanAll) @ (Mean2 - MeanAll).T
    )

    # Spatial WITHIN-CLASS matrix (plus regularization)
    Sw = p1 * Covavg1 + p2 * Covavg2
    Sw = (1 - th) * Sw + th * np.eye(n_chan)

    # Solve the generalized eigenvalue problem
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(-eigvals)
    V1 = np.diag(eigvals[sorted_indices])   # Ordered eigenvalues (diagonal)
    U1 = eigvecs[:, sorted_indices]         # Ordered eigenvectors (filters)

    return U1, V1

def FCB_projections(z1, z2, U):
    """
    Project EEG data onto spatial filters (FCB projections).

    Adapted from: https://github.com/gpiresML/FCB-spatial-filter
    Original author: Gabriel Pires, 2022

    Tidied by Rosie Zheng, 2025

    Args:
    z1, z2 : np.array
        EEG data arrays for class 1 and class 2, shape = (channels, samples, trials)
    U : np.array
        Spatial filters (eigenvectors), shape = (channels, n_filters)

    Returns:
    z1_f, z2_f : np.array
        Projected data for class 1 and 2, shape = (n_filters, samples, trials)
    """
    n_chan, n_samples, n_trials_1 = z1.shape
    n_filters = U.shape[1]
    n_trials_2 = z2.shape[2]

    # Initialize arrays for projected data
    z1_f = np.zeros((n_filters, n_samples, n_trials_1))
    z2_f = np.zeros((n_filters, n_samples, n_trials_2))

    # Project each trial of class 1
    for i in range(n_trials_1):
        z1_f[:, :, i] = U.T @ z1[:, :, i]

    # Project each trial of class 2
    for i in range(n_trials_2):
        z2_f[:, :, i] = U.T @ z2[:, :, i]

    return z1_f, z2_f

def rsquare(q, r):
    """
    Compute the r^2 (coefficient of determination) between two signals.

    Adapted from: https://github.com/gpiresML/FCB-spatial-filter
    Original author: Gabriel Pires, 2022

    Tidied by Rosie Zheng, 2025

    Args:
    q, r : np.array
        Input arrays, shape = (samples, channels) or similar

    Returns:
    erg : float
        r^2 value
    """
    sum1 = np.sum(q)
    sum2 = np.sum(r)
    n1 = q.shape[0]
    n2 = r.shape[0]
    sumsqu1 = np.sum(q * q)
    sumsqu2 = np.sum(r * r)

    G = ((sum1 + sum2) ** 2) / (n1 + n2)
    numerator = (sum1 ** 2) / n1 + (sum2 ** 2) / n2 - G
    denominator = sumsqu1 + sumsqu2 - G

    erg = numerator / denominator
    return erg

def plot_rsquare(t, ressq):
    """
    Plot r^2 values as a color map over time and channels.

    Adapted from: https://github.com/gpiresML/FCB-spatial-filter
    Original author: Gabriel Pires, 2022

    Tidied by Rosie Zheng, 2025

    Args:
    t : np.array
        Time vector
    ressq : np.array
        r^2 values, shape = (time, channels)
    """
    data2plot = ressq.T
    n_channels, n_time = data2plot.shape

    # Add zero padding for pcolormesh
    data2plot = np.concatenate((data2plot, np.zeros((n_channels, 1))), axis=1)
    data2plot = np.concatenate((data2plot, np.zeros((1, data2plot.shape[1]))), axis=0)

    # Extend time axis
    x_data = np.append(t, t[-1] + (t[-1] - t[-2]))

    # Dynamisch kleurenspectrum op basis van je data
    vmin = np.min(ressq)
    vmax = np.max(ressq)

    plt.pcolormesh(x_data, np.arange(n_channels + 1), data2plot, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    plt.colorbar(label="r$^2$")
    plt.ylabel("Channels")
    plt.xlabel("Time (s)")
    plt.title("Statistical r$^2$ between class1 and class2")
    plt.tight_layout()
    plt.show()

def save_crossval_results(function_name, metrics, fold_means, fold_stds, params, save_path="results/"):
    """
    Save cross-validation results, summary statistics, and parameters to a JSON file for documentation and reproducibility.

    Args:
        function_name (str): Name of the cross-validation or training function used
        metrics (dict): Pooled metrics (across all folds)
        fold_means (dict): Dictionary of metric means across folds
        fold_stds (dict): Dictionary of metric standard deviations across folds
        params (dict): Dictionary of parameters/settings used for this run
        save_path (str): Directory where the result JSON file should be saved

    Returns:
        filename (str): Path to the saved JSON file
    """

    # Build result dictionary with metadata, metrics, and parameters
    result = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),   # Timestamp for traceability
        "function": function_name,                                  # Function name or method identifier
        "parameters": params,                                       # All experiment parameters
        "pooled_metrics": to_serializable(metrics),                 # Pooled metrics, made serializable if needed
        "fold_means": to_serializable(fold_means),                  # Fold-wise means, made serializable if needed
        "fold_stds": to_serializable(fold_stds),                    # Fold-wise stds, made serializable if needed
    }

    # Build a unique filename with parameters and timestamp for tracking runs
    param_str = "_".join(f"{k}-{v}" for k, v in params.items())
    filename = f"{save_path}{function_name}_{param_str}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    os.makedirs(save_path, exist_ok=True)

    with open(filename, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved results to {filename}")
    return filename

def crossval_metrics_stratified_kfold(X, y, n_splits=5, plot_roc=False, random_state=42):
    """
    Run stratified K-Fold cross-validation with LDA and return pooled and per-fold metrics.

    Args:
        X (np.ndarray): Feature matrix [n_samples, n_features]
        y (np.ndarray): Labels [n_samples]
        n_splits (int): Number of cross-validation folds
        plot_roc (bool): If True, plot ROC curve using pooled predictions
        random_state (int): Seed for reproducibility

    Returns:
        pooled_metrics (dict): Overall metrics (across all splits)
        fold_means (dict): Mean metrics per fold
        fold_stds (dict): Std of metrics per fold
    """

    # Initialize classifier (LDA with shrinkage, suitable for EEG)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto")
    # Create stratified splits to preserve label proportions in each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_y_true, all_y_pred, all_y_score = [], [], []
    fold_metrics = []

    # Cross-validation loop
    for train_idx, test_idx in skf.split(X, y):
        # Train on current fold's training set
        clf.fit(X[train_idx], y[train_idx])
        # Predict labels on current fold's test set
        y_pred = clf.predict(X[test_idx])
        y_true = y[test_idx]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        # If classifier supports predict_proba, get probability for ROC/AUC
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X[test_idx])[:, 1]
            all_y_score.extend(y_score)
        # Collect per-fold metrics
        fold_metrics.append({
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        })

    # Convert lists to arrays for metric calculation
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_score = np.array(all_y_score)

    # Compute metrics pooled over all test samples in all folds
    pooled_metrics = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'balanced_accuracy': balanced_accuracy_score(all_y_true, all_y_pred),
        'precision': precision_score(all_y_true, all_y_pred),
        'recall': recall_score(all_y_true, all_y_pred),
        'f1': f1_score(all_y_true, all_y_pred),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred)
    }

    # If binary classification, also compute ROC-AUC and optionally plot ROC curve
    if len(np.unique(all_y_true)) == 2:
        auc = roc_auc_score(all_y_true, all_y_score)
        pooled_metrics['roc_auc'] = auc
        if plot_roc:
            fpr, tpr, thresholds = roc_curve(all_y_true, all_y_score)
            plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()

    # Compute mean and std of each metric across folds
    fold_means = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    fold_stds = {k: np.std([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    return pooled_metrics, fold_means, fold_stds

def crossval_metrics_leave_one_group(X, y, groups, plot_roc=False):
    """
    Perform Leave-One-Group-Out (LOGO) cross-validation with LDA classifier
    and compute pooled and per-fold metrics.

    Each unique value in `groups` defines a group (e.g., subject or session).
    For each fold, one group is used as the test set and the rest as the train set.

    Args:
        X (np.ndarray): Feature matrix of shape [n_samples, n_features]
        y (np.ndarray): Labels array of shape [n_samples]
        groups (array-like): Group labels of shape [n_samples], one per sample
        plot_roc (bool): If True, plots ROC curve based on pooled test scores

    Returns:
        pooled_metrics (dict): Overall metrics across all folds
        fold_means (dict): Mean of metrics over folds
        fold_stds (dict): Standard deviation of metrics over folds
    """
    # Initialize LeaveOneGroupOut splitter and classifier (LDA with shrinkage)
    logo = LeaveOneGroupOut()
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto")

    all_y_true, all_y_pred, all_y_score = [], [], []
    fold_metrics = []

    # Iterate over each group as the test set
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        # Train LDA classifier on all groups except the test group
        clf.fit(X[train_idx], y[train_idx])
        # Predict on left-out group
        y_pred = clf.predict(X[test_idx])
        y_true = y[test_idx]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        # Get predicted probabilities for ROC/AUC (if available)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X[test_idx])[:, 1]
            all_y_score.extend(y_score)
        # Store per-fold metrics
        fold_metrics.append({
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        })

    # Convert lists to arrays for consistent metric computation
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_score = np.array(all_y_score)

    # Compute overall (pooled) metrics across all test predictions
    pooled_metrics = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'balanced_accuracy': balanced_accuracy_score(all_y_true, all_y_pred),
        'precision': precision_score(all_y_true, all_y_pred),
        'recall': recall_score(all_y_true, all_y_pred),
        'f1': f1_score(all_y_true, all_y_pred),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred)
    }

    # If binary classification, also compute ROC-AUC and optionally plot ROC curve
    if len(np.unique(all_y_true)) == 2:
        auc = roc_auc_score(all_y_true, all_y_score)
        pooled_metrics['roc_auc'] = auc
        if plot_roc:
            fpr, tpr, thresholds = roc_curve(all_y_true, all_y_score)
            plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()

    # Calculate mean and std for each metric over all folds
    fold_means = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    fold_stds = {k: np.std([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}

    return pooled_metrics, fold_means, fold_stds

def crossval_metrics_leave_one_session(X, y, subject_ids, session_ids, plot_roc=False):
    """
    Perform Leave-One-Session-Out cross-validation using unique subject-session pairs as groups.
    Each epoch is assigned a group label of the form 'subject_session'.
    For each fold, all epochs from one subject-session pair are used as the test set.

    Args:
        X (np.ndarray): Feature matrix of shape [n_samples, n_features]
        y (np.ndarray): Labels array of shape [n_samples]
        subject_ids (array-like): Subject identifier for each epoch [n_samples]
        session_ids (array-like): Session identifier for each epoch [n_samples]
        plot_roc (bool): If True, plots ROC curve for pooled predictions

    Returns:
        pooled_metrics (dict): Overall metrics pooled across all folds
        fold_means (dict): Mean of metrics over all folds
        fold_stds (dict): Standard deviation of metrics over all folds
    """
    # Combine subject and session into a unique group string for each epoch
    group_ids = np.array([f"{subj}_{sess}" for subj, sess in zip(subject_ids, session_ids)])

    # Call LOGO cross-validation using these session-group labels
    return crossval_metrics_leave_one_group(X, y, group_ids, plot_roc=plot_roc)

def to_serializable(obj):
    """
    Recursively convert numpy arrays (and numpy scalars) inside nested dicts, lists, or tuples to native Python lists or types.
    This enables safe JSON serialization of results containing numpy types.

    Args:
        obj: Object to be converted (can be dict, list, tuple, numpy array, or scalar)

    Returns:
        Native Python object (with all numpy types converted to built-ins)
    """
    # If dictionary: recursively convert each value
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    # If list: recursively convert each element
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    # If tuple: recursively convert each element and reconstruct tuple
    elif isinstance(obj, tuple):
        return tuple(to_serializable(v) for v in obj)
    # If numpy array or scalar (has "tolist" method): convert to list or native type
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    # Otherwise: return as is (should be JSON-serializable)
    else:
        return obj
    
class EEGGenerator(nn.Module):
    """
    Generator network for synthetic EEG data using a 1D convolutional transpose architecture.

    Maps a latent noise vector to a synthetic EEG epoch with shape [batch, n_time, n_features].
    Architecture: latent vector --> fully-connected projection --> sequence of ConvTranspose1d upsampling blocks.

    Args:
        latent_dim (int): Dimensionality of input latent noise vector
        n_time (int): Number of time points in output EEG epoch
        n_features (int): Number of features (channels) in output EEG epoch
    """
    def __init__(self, latent_dim, n_time, n_features):
        super().__init__()
        self.n_time = n_time
        self.n_features = n_features

        # Project latent vector up to a small feature map [batch, 128*(n_time//8)]
        self.proj = nn.Linear(latent_dim, 128 * (n_time // 8))

        # Sequential upsampling with ConvTranspose1d layers
        # Input shape: (batch, 128, n_time//8)
        self.net = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (batch, 64, n_time//4)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # Output: (batch, 32, n_time//2)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(32, n_features, kernel_size=4, stride=2, padding=1),  # Output: (batch, n_features, n_time)
            nn.Tanh(),  # Clamp output to [-1, 1] range
        )

    def forward(self, z):
        """
        Forward pass: Generate a batch of synthetic EEG epochs from latent noise.

        Args:
            z (torch.Tensor): Input noise vector of shape [batch, latent_dim]

        Returns:
            torch.Tensor: Synthetic EEG epochs, shape [batch, n_time, n_features]
        """
        x = self.proj(z)  # Fully connected projection to [batch, 128 * (n_time//8)]
        x = x.view(z.size(0), 128, self.n_time // 8)  # Reshape for ConvTranspose1d: (batch, 128, n_time//8)
        x = self.net(x)  # Upsample to (batch, n_features, n_time)
        x = x.permute(0, 2, 1)  # Rearrange to (batch, n_time, n_features)
        return x

class EEGDiscriminator(nn.Module):
    """
    Discriminator network for EEG GAN. 
    Evaluates whether an input EEG epoch is real or synthetic.

    Architecture:
    - Stacked 1D convolutional layers with increasing feature depth.
    - Flattened output passed through a fully connected layer.
    - Output is a probability (after sigmoid): real=1, fake=0.

    Args:
        n_time (int): Number of time points in the input EEG epoch
        n_features (int): Number of features (channels) in the input EEG epoch
    """

    def __init__(self, n_time, n_features):
        super().__init__()
        # Main feature extraction stack: 1D conv layers, batch norm, activations
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=4, stride=2, padding=1),   # Downsample time axis by 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),           # Downsample by 2 again
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),          # Downsample by 2 again
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),                                                    # Flatten all features for FC layer
            nn.Linear((n_time // 8) * 128, 1),                               # Fully connected: downsampled time * channels
            nn.Sigmoid(),                                                    # Output probability (real/fake)
        )

    def forward(self, x):
        """
        Forward pass: classify a batch of EEG epochs as real or fake.

        Args:
            x (torch.Tensor): Input of shape [batch, n_time, n_features]

        Returns:
            torch.Tensor: Output probabilities, shape [batch, 1]
        """
        x = x.permute(0, 2, 1)   # Rearrange to [batch, n_features, n_time] for Conv1d
        return self.net(x)        # Output: probability [batch, 1]

def train_gan(error_epochs, n_epochs=2000, latent_dim=32, batch_size=32, device=None):
    """
    Train a simple GAN to model the distribution of error EEG epochs.

    Args:
        error_epochs (np.ndarray): Array of error trials, shape [n_trials, n_time, n_features]
        n_epochs (int): Number of training epochs for GAN
        latent_dim (int): Dimensionality of generator input noise
        batch_size (int): Mini-batch size
        device (torch.device or None): Target device

    Returns:
        generator (EEGGenerator): Trained generator
        err_mean (np.ndarray): Mean used for z-score normalization (for inverse scaling)
        err_std (np.ndarray): Std used for z-score normalization (for inverse scaling)
    """

    # Extract time and feature dimensions
    n_time, n_features = error_epochs.shape[1], error_epochs.shape[2]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Z-score normalize per channel
    err_mean = error_epochs.mean(axis=(0,1), keepdims=True)
    err_std = error_epochs.std(axis=(0,1), keepdims=True)
    error_epochs_scaled = (error_epochs - err_mean) / (err_std + 1e-6)
    error_tensor = torch.tensor(error_epochs_scaled, dtype=torch.float32).to(device)

    generator = EEGGenerator(latent_dim, n_time, n_features).to(device)
    discriminator = EEGDiscriminator(n_time, n_features).to(device)

    # Adam optimizers for both generator and discriminator
    opt_g = optim.Adam(generator.parameters(), lr=2e-4)
    opt_d = optim.Adam(discriminator.parameters(), lr=2e-4)

    criterion = nn.BCELoss()

    for epoch in range(n_epochs):
        # Randomly sample a mini-batch of real error epochs
        idx = np.random.randint(0, error_tensor.shape[0], batch_size)
        real = error_tensor[idx]
        valid = torch.ones(batch_size, 1, device=device)   # Label for real data
        fake = torch.zeros(batch_size, 1, device=device)   # Label for fake data

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim, device=device)     # Sample random noise
        gen_data = generator(z)                                    # Generate synthetic error epochs

        d_real = discriminator(real)                               # Discriminator on real data
        d_fake = discriminator(gen_data.detach())                  # Discriminator on generated data

        # Discriminator loss: classify real as 1, fake as 0
        loss_d = criterion(d_real, valid) + criterion(d_fake, fake)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # Train Generator
        d_fake = discriminator(gen_data)                           # Re-evaluate fake data, now update G
        loss_g = criterion(d_fake, valid)                          # Generator wants D to classify fake as real (label=1)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss_D {loss_d.item():.4f} Loss_G {loss_g.item():.4f}")

    return generator, err_mean, err_std

def generate_synthetic_errors(generator, n_samples, latent_dim, n_time, n_features, err_mean, err_std, device=None):
    """
    Generate synthetic EEG error epochs using a trained GAN generator.

    Args:
        generator (EEGGenerator): Trained generator model
        n_samples (int): Number of synthetic epochs to generate
        latent_dim (int): Dimensionality of generator input noise
        n_time (int): Number of time points per epoch
        n_features (int): Number of features (channels) per epoch
        err_mean (np.ndarray): Mean used for z-score normalization during GAN training
        err_std (np.ndarray): Std used for z-score normalization during GAN training
        device (torch.device or None): Target device

    Returns:
        synth (np.ndarray): Synthetic error epochs, shape [n_samples, n_time, n_features]
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval()  
    with torch.no_grad():
        # Sample random noise as input for generator
        z = torch.randn(n_samples, latent_dim, device=device)
        # Generate synthetic data (still z-scored)
        synth = generator(z).cpu().numpy()
    # Reverse z-score normalization to restore original scale
    synth = synth * (err_std + 1e-6) + err_mean
    return synth

def save_augmented_data(X_augmented, y_augmented, fold, network, method, save_dir='augmented_data'):
    """
    Save augmented data and labels to .npy files for reproducibility and later use.

    Args:
        X_augmented (np.ndarray): Augmented input data array
        y_augmented (np.ndarray): Corresponding labels array
        fold (int or str): Fold index or identifier (used in filename)
        network (str): Name of the network/model (e.g., 'CNN', 'Transformer')
        method (str): Augmentation or training method name (e.g., 'GAN')
        save_dir (str): Directory where files will be saved

    Returns:
        None. Files are saved to disk.
    """

    os.makedirs(save_dir, exist_ok=True)

    fname_X = os.path.join(save_dir, f'Xaug_{fold}_{network}_{method}.npy')
    fname_y = os.path.join(save_dir, f'yaug_{fold}_{network}_{method}.npy')

    np.save(fname_X, X_augmented)
    np.save(fname_y, y_augmented)

    print(f"Saved: {fname_X} and {fname_y}")

class EEG_Transformer(nn.Module):
    """
    Transformer encoder model for EEG sequence classification.

    The model projects multi-channel EEG data into an embedding space,
    processes it with stacked self-attention layers, then classifies each sequence.

    Args:
        n_features (int): Number of input features per time point (e.g., EEG channels)
        n_time (int): Number of time points per trial/epoch
        d_model (int): Dimensionality of embeddings in transformer
        nhead (int): Number of attention heads
        num_layers (int): Number of stacked transformer encoder layers
        dim_feedforward (int): Size of the feedforward layer in the transformer
        dropout (float): Dropout rate

    Usage:
        model = EEG_Transformer(n_features, n_time)
        output = model(x)  # x shape: [batch, 1, time, features]
    """

    def __init__(self, n_features, n_time, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        # Stack multiple encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier: flatten over time, map through MLP, output single logit
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # Flatten sequence to vector [batch, d_model * n_time]
            nn.Linear(d_model * n_time, 64),         
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)                         # Output logit for binary classification
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor, shape [batch, 1, time, features]

        Returns:
            torch.Tensor: Output logits, shape [batch]
        """
        x = x.squeeze(1)               
        x = self.input_proj(x)        
        x = self.transformer(x)        
        out = self.classifier(x)        
        return out.squeeze(-1)           # Return logits, shape [batch]
 
def crossval_gan_augmented_transformer(
    correct_epochs, error_epochs, correct_labels, error_labels,
    n_splits=5, latent_dim=32, n_gan_epochs=1000,
    transformer_epochs=20, batch_size=32, lr=1e-3, plot_roc=False, random_state=42
):
    """
    Stratified K-Fold cross-validation for EEG Transformer classifier, with GAN-based augmentation for class balancing.

    - For each fold, GAN augmentation is applied only to the training set to synthetically balance error/correct epochs.
    - After augmentation, a Transformer is trained and evaluated on the held-out fold.
    - Metrics are aggregated over all folds.

    Args:
        correct_epochs (np.ndarray): Correct-labeled EEG epochs, shape [n_trials, n_time, n_features]
        error_epochs (np.ndarray): Error-labeled EEG epochs, shape [n_trials, n_time, n_features]
        correct_labels (np.ndarray): Labels for correct epochs (usually zeros)
        error_labels (np.ndarray): Labels for error epochs (usually ones)
        n_splits (int): Number of cross-validation folds
        latent_dim (int): Latent dimension for GAN generator input
        n_gan_epochs (int): Number of GAN training epochs per fold
        transformer_epochs (int): Number of Transformer training epochs per fold
        batch_size (int): Batch size for GAN and Transformer training
        lr (float): Learning rate for optimizer
        plot_roc (bool): Whether to plot ROC curve after evaluation
        random_state (int): Random seed for reproducibility

    Returns:
        pooled_metrics (dict): Pooled metrics over all folds
        fold_means (dict): Mean metrics across folds
        fold_stds (dict): Std of metrics across folds
    """

    # Combine all epochs and labels for stratified splitting
    all_X = np.concatenate([correct_epochs, error_epochs], axis=0)
    all_y = np.concatenate([correct_labels, error_labels], axis=0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_y_true, all_y_pred, all_y_score = [], [], []
    fold_metrics = []
    n_time, n_features = correct_epochs.shape[1], correct_epochs.shape[2]
    fold = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for train_idx, test_idx in skf.split(all_X, all_y):
        # Split into train/test folds
        X_train, X_test = all_X[train_idx], all_X[test_idx]
        y_train, y_test = all_y[train_idx], all_y[test_idx]

        # Augment training data with GAN if needed
        # Split training set into correct and error epochs
        train_correct_mask = y_train == 0
        train_error_mask = y_train == 1
        train_correct_epochs = X_train[train_correct_mask]
        train_error_epochs = X_train[train_error_mask]
        # Determine number of synthetic error samples needed to balance
        n_synth = len(train_correct_epochs) - len(train_error_epochs)
        if n_synth > 0:
            # Train GAN on training set error epochs only
            generator, err_mean, err_std = train_gan(
                train_error_epochs, n_epochs=n_gan_epochs, latent_dim=latent_dim, device=device)
            # Generate synthetic error epochs for class balancing
            synthetic_error_epochs = generate_synthetic_errors(
                generator, n_samples=n_synth, latent_dim=latent_dim,
                n_time=n_time, n_features=n_features, err_mean=err_mean, err_std=err_std, device=device)
            # Concatenate real and synthetic error epochs
            train_error_aug = np.concatenate([train_error_epochs, synthetic_error_epochs], axis=0)
        else:
            # No augmentation needed if already balanced
            train_error_aug = train_error_epochs

        # Create label arrays for augmented training set
        train_error_labels = np.ones(train_error_aug.shape[0])
        train_correct_labels = np.zeros(train_correct_epochs.shape[0])
        X_train_aug = np.concatenate([train_correct_epochs, train_error_aug], axis=0)
        y_train_aug = np.concatenate([train_correct_labels, train_error_labels], axis=0)

        # Shuffle augmented training set to mix error/correct trials
        perm = np.random.permutation(len(y_train_aug))
        X_train_aug = X_train_aug[perm]
        y_train_aug = y_train_aug[perm]

        # Optionally save augmented data for reproducibility
        save_augmented_data(X_train_aug, y_train_aug, fold, 'transformer', 'skf')

        # Prepare data loaders for PyTorch
        X_train_torch = torch.tensor(X_train_aug, dtype=torch.float32).unsqueeze(1)  # [batch, 1, time, features]
        y_train_torch = torch.tensor(y_train_aug, dtype=torch.float32)
        X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        y_test_torch = torch.tensor(y_test, dtype=torch.float32)
        train_ds = TensorDataset(X_train_torch, y_train_torch)
        test_ds = TensorDataset(X_test_torch, y_test_torch)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Compute class weights for imbalanced loss (helps rare classes)
        n_pos = (y_train_aug == 1).sum().item()
        n_neg = (y_train_aug == 0).sum().item()
        pos_weight = torch.tensor([n_neg / n_pos]).to(device) if n_pos > 0 else torch.tensor([1.0]).to(device)

        # Initialize and train Transformer classifier
        model = EEG_Transformer(n_features=n_features, n_time=n_time).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for ep in range(transformer_epochs):
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        # Evaluate on test fold
        model.eval()
        y_pred_fold, y_true_fold, y_score_fold = [], [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                logits = model(xb)
                y_score = torch.sigmoid(logits).cpu().numpy()
                y_pred = (y_score > 0.5).astype(int)
                y_pred_fold.extend(y_pred)
                y_true_fold.extend(yb.cpu().numpy())
                y_score_fold.extend(y_score)
        all_y_true.extend(y_true_fold)
        all_y_pred.extend(y_pred_fold)
        all_y_score.extend(y_score_fold)

        # Compute and store metrics for this fold
        fold_metrics.append({
            'accuracy': accuracy_score(y_true_fold, y_pred_fold),
            'f1': f1_score(y_true_fold, y_pred_fold),
            'precision': precision_score(y_true_fold, y_pred_fold),
            'recall': recall_score(y_true_fold, y_pred_fold),
            'balanced_accuracy': balanced_accuracy_score(y_true_fold, y_pred_fold)
        })
        fold += 1

    # Aggregate and report results over all folds
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_score = np.array(all_y_score)

    # Compute overall metrics
    pooled_metrics = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'balanced_accuracy': balanced_accuracy_score(all_y_true, all_y_pred),
        'precision': precision_score(all_y_true, all_y_pred),
        'recall': recall_score(all_y_true, all_y_pred),
        'f1': f1_score(all_y_true, all_y_pred),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred)
    }
    # Compute ROC-AUC if binary classification
    if len(np.unique(all_y_true)) == 2:
        auc = roc_auc_score(all_y_true, all_y_score)
        pooled_metrics['roc_auc'] = auc
        if plot_roc:
            fpr, tpr, thresholds = roc_curve(all_y_true, all_y_score)
            plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve (GAN-augmented Transformer)')
            plt.legend()
            plt.show()

    # Mean and std over folds for each metric
    fold_means = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    fold_stds = {k: np.std([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    return pooled_metrics, fold_means, fold_stds

def crossval_gan_augmented_transformer_logo(
    correct_epochs, error_epochs, correct_labels, error_labels,
    group_labels,  # array-like, same length as all_X
    latent_dim=32, n_gan_epochs=1000,
    transformer_epochs=20, batch_size=32, lr=1e-3, plot_roc=False, random_state=42
):
    """
    Leave-One-Group-Out cross-validation for EEG Transformer classifier with GAN-based error trial augmentation.

    - For each fold, one group (subject/session) is held out for validation; the rest are used for training.
    - GAN is trained only on error trials from the training set to balance error/correct classes.
    - Metrics are aggregated across all groups.

    Args:
        correct_epochs (np.ndarray): Correct-labeled EEG epochs, shape [n_trials, n_time, n_features]
        error_epochs (np.ndarray): Error-labeled EEG epochs, shape [n_trials, n_time, n_features]
        correct_labels (np.ndarray): Labels for correct epochs (typically zeros)
        error_labels (np.ndarray): Labels for error epochs (typically ones)
        group_labels (array-like): Group IDs for all epochs (one per trial; e.g., subject or session)
        latent_dim (int): Latent dimension for GAN generator input
        n_gan_epochs (int): GAN training epochs
        transformer_epochs (int): Transformer training epochs per fold
        batch_size (int): Batch size for GAN/Transformer
        lr (float): Learning rate
        plot_roc (bool): Plot ROC curve after evaluation
        random_state (int): For reproducibility

    Returns:
        pooled_metrics (dict): Metrics pooled over all folds
        fold_means (dict): Mean metrics across folds
        fold_stds (dict): Std of metrics across folds
    """
    # Concatenate epochs and labels for splitting
    all_X = np.concatenate([correct_epochs, error_epochs], axis=0)
    all_y = np.concatenate([correct_labels, error_labels], axis=0)
    all_groups = np.asarray(group_labels)

    logo = LeaveOneGroupOut()  # Each group left out once as test set
    all_y_true, all_y_pred, all_y_score = [], [], []
    fold_metrics = []
    n_time, n_features = correct_epochs.shape[1], correct_epochs.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loop over all possible group splits
    for train_idx, test_idx in logo.split(all_X, all_y, groups=all_groups):
        # Split into training and held-out (test) set by group
        X_train, X_test = all_X[train_idx], all_X[test_idx]
        y_train, y_test = all_y[train_idx], all_y[test_idx]

        # Split training data into correct and error trials
        train_correct_mask = y_train == 0
        train_error_mask = y_train == 1
        train_correct_epochs = X_train[train_correct_mask]
        train_error_epochs = X_train[train_error_mask]

        # Compute required number of synthetic error samples to balance
        n_synth = len(train_correct_epochs) - len(train_error_epochs)
        if n_synth > 0:
            # Train GAN using only training set error trials
            generator, err_mean, err_std = train_gan(
                train_error_epochs, n_epochs=n_gan_epochs, latent_dim=latent_dim, device=device)
            # Generate synthetic error epochs to balance the classes
            synthetic_error_epochs = generate_synthetic_errors(
                generator, n_samples=n_synth, latent_dim=latent_dim,
                n_time=n_time, n_features=n_features, err_mean=err_mean, err_std=err_std, device=device)
            # Combine real and synthetic error epochs
            train_error_aug = np.concatenate([train_error_epochs, synthetic_error_epochs], axis=0)
        else:
            # No augmentation if already balanced
            train_error_aug = train_error_epochs

        # Prepare labels for augmented training set
        train_error_labels = np.ones(train_error_aug.shape[0])
        train_correct_labels = np.zeros(train_correct_epochs.shape[0])
        X_train_aug = np.concatenate([train_correct_epochs, train_error_aug], axis=0)
        y_train_aug = np.concatenate([train_correct_labels, train_error_labels], axis=0)

        # Shuffle training data (important for SGD)
        perm = np.random.permutation(len(y_train_aug))
        X_train_aug = X_train_aug[perm]
        y_train_aug = y_train_aug[perm]

        # Train Transformer on augmented training set
        X_train_torch = torch.tensor(X_train_aug, dtype=torch.float32).unsqueeze(1)  # [batch, 1, time, features]
        y_train_torch = torch.tensor(y_train_aug, dtype=torch.float32)
        X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        y_test_torch = torch.tensor(y_test, dtype=torch.float32)
        train_ds = TensorDataset(X_train_torch, y_train_torch)
        test_ds = TensorDataset(X_test_torch, y_test_torch)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Compute positive class weight for BCE loss (helps if still imbalanced)
        n_pos = (y_train_aug == 1).sum().item()
        n_neg = (y_train_aug == 0).sum().item()
        pos_weight = torch.tensor([n_neg / n_pos]).to(device) if n_pos > 0 else torch.tensor([1.0]).to(device)

        # Initialize Transformer model and optimizer
        model = EEG_Transformer(n_features=n_features, n_time=n_time).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for ep in range(transformer_epochs):
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        # Evaluate on held-out group
        model.eval()
        y_pred_fold, y_true_fold, y_score_fold = [], [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                logits = model(xb)
                y_score = torch.sigmoid(logits).cpu().numpy()
                y_pred = (y_score > 0.5).astype(int)
                y_pred_fold.extend(y_pred)
                y_true_fold.extend(yb.cpu().numpy())
                y_score_fold.extend(y_score)
        all_y_true.extend(y_true_fold)
        all_y_pred.extend(y_pred_fold)
        all_y_score.extend(y_score_fold)
        # Store metrics for this group/fold
        fold_metrics.append({
            'accuracy': accuracy_score(y_true_fold, y_pred_fold),
            'f1': f1_score(y_true_fold, y_pred_fold),
            'precision': precision_score(y_true_fold, y_pred_fold),
            'recall': recall_score(y_true_fold, y_pred_fold),
            'balanced_accuracy': balanced_accuracy_score(y_true_fold, y_pred_fold)
        })

    # Aggregate metrics over all groups/folds
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_score = np.array(all_y_score)
    pooled_metrics = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'balanced_accuracy': balanced_accuracy_score(all_y_true, all_y_pred),
        'precision': precision_score(all_y_true, all_y_pred),
        'recall': recall_score(all_y_true, all_y_pred),
        'f1': f1_score(all_y_true, all_y_pred),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred)
    }
    # ROC-AUC calculation for binary classification
    if len(np.unique(all_y_true)) == 2:
        auc = roc_auc_score(all_y_true, all_y_score)
        pooled_metrics['roc_auc'] = auc
        if plot_roc:
            fpr, tpr, thresholds = roc_curve(all_y_true, all_y_score)
            plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve (GAN-augmented Transformer, LOGO-CV)')
            plt.legend()
            plt.show()
    # Compute means and stds for metrics across all folds
    fold_means = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    fold_stds = {k: np.std([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    return pooled_metrics, fold_means, fold_stds


