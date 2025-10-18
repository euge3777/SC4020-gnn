import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, MultiheadAttention, LayerNorm
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, degree
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv, TransformerConv
from torch_geometric.transforms import LaplacianLambdaMax
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from node2vec import Node2Vec
import networkx as nx
from scipy.sparse import coo_matrix
import warnings
import itertools
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class CoraDataLoader:
    def __init__(self, dataset_name="Cora"):
        self.dataset_name = dataset_name
        
    def load_data(self):
        """Load Cora dataset from PyTorch Geometric"""
        print(f"Loading {self.dataset_name} dataset...")
        
        # Load dataset with normalization
        transform = T.NormalizeFeatures()
        dataset = Planetoid(root='data/Planetoid', name=self.dataset_name, transform=transform)
        
        data = dataset[0]
        
        print(f"Dataset statistics:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Features: {data.num_node_features}")
        print(f"  Classes: {dataset.num_classes}")
        print(f"  Training nodes: {data.train_mask.sum().item()}")
        print(f"  Validation nodes: {data.val_mask.sum().item()}")
        print(f"  Test nodes: {data.test_mask.sum().item()}")
        
        return data, dataset.num_classes

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2, 
                 dropout=0.5, attn_dropout=0.0, alpha=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                                 dropout=attn_dropout, negative_slope=alpha))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, 
                                     dropout=attn_dropout, negative_slope=alpha))
        
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, 
                                 dropout=attn_dropout, negative_slope=alpha))
        
    def forward(self, x, edge_index, return_attention_weights=False):
        attention_weights = []
        
        for i, conv in enumerate(self.convs[:-1]):
            if return_attention_weights:
                x, (edge_index_att, alpha) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_index_att, alpha))
            else:
                x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
            x, (edge_index_att, alpha) = self.convs[-1](x, edge_index, return_attention_weights=True)
            attention_weights.append((edge_index_att, alpha))
        else:
            x = self.convs[-1](x, edge_index)
        
        x = F.log_softmax(x, dim=1)
        
        if return_attention_weights:
            return x, attention_weights
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.5, aggregator='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, 
                 gcn_layers=1, transformer_layers=2, dropout=0.5, attn_dropout=0.0,
                 pre_norm=True, residual_dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.gcn_layers = gcn_layers
        self.transformer_layers = transformer_layers
        self.pre_norm = pre_norm
        self.residual_dropout = residual_dropout
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() if pre_norm else None
        
        current_channels = in_channels
        
        # GCN layers
        for i in range(gcn_layers):
            self.convs.append(GCNConv(current_channels, hidden_channels))
            if pre_norm and i > 0:  # Don't add norm before first layer
                self.norms.append(torch.nn.LayerNorm(current_channels))
            current_channels = hidden_channels
        
        # Transformer layers
        for i in range(transformer_layers):
            if i == transformer_layers - 1:  # Last layer
                self.convs.append(TransformerConv(current_channels, out_channels, heads=1, 
                                                concat=False, dropout=attn_dropout))
                # No norm after final layer
            else:
                self.convs.append(TransformerConv(current_channels, hidden_channels, heads=heads, 
                                                dropout=attn_dropout))
                if pre_norm:
                    self.norms.append(torch.nn.LayerNorm(current_channels))
                current_channels = hidden_channels * heads
        
    def forward(self, x, edge_index):
        layer_idx = 0
        norm_idx = 0
        
        # GCN layers
        for i in range(self.gcn_layers):
            # Apply normalization before layer (except first layer)
            if self.pre_norm and i > 0 and norm_idx < len(self.norms):
                x = self.norms[norm_idx](x)
                norm_idx += 1
            
            # Apply convolution
            x = self.convs[layer_idx](x, edge_index)
            layer_idx += 1
            
            # Apply activation and dropout (except last layer)
            if i < self.gcn_layers - 1 or self.transformer_layers > 0:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Transformer layers
        for i in range(self.transformer_layers):
            # Apply normalization before layer (except after GCN if no norm there)
            if self.pre_norm and (self.gcn_layers > 0 or i > 0) and norm_idx < len(self.norms):
                x = self.norms[norm_idx](x)
                norm_idx += 1
            
            # Apply convolution
            x = self.convs[layer_idx](x, edge_index)
            layer_idx += 1
            
            # Apply activation and dropout (except last layer)
            if i < self.transformer_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=1)

class AdvancedTrainer:
    def __init__(self, model, data, lr=0.01, weight_decay=5e-4, grad_clip_norm=None):
        self.model = model.to(DEVICE)
        self.data = data.to(DEVICE)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip_norm = grad_clip_norm
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_micro_f1_scores = []
        self.test_accuracies = []
        self.test_micro_f1s = []
        self.mse_scores = []
        
        # Early stopping metrics
        self.early_stop_epoch = None
        self.early_stop_train_loss = None
        self.early_stop_val_f1 = None
        
    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        
        # Gradient clipping if specified
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            
            y_true = self.data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            y_prob = torch.softmax(out[mask], dim=1).cpu().numpy()
            
            micro_f1 = f1_score(y_true, y_pred, average='micro')
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate MSE between predicted probabilities and one-hot encoded true labels
            y_true_onehot = np.eye(out.size(1))[y_true]
            mse = mean_squared_error(y_true_onehot, y_prob)
            
            # Calculate validation loss
            val_loss = F.nll_loss(out[mask], self.data.y[mask]).item()
            
            return micro_f1, macro_f1, accuracy, mse, val_loss, y_true, y_pred
    
    def train(self, epochs=200, patience=50, verbose=True):
        best_val_f1 = 0
        patience_counter = 0
        
        if verbose:
            print(f"Training for up to {epochs} epochs with patience {patience}...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation evaluation
            val_micro_f1, val_macro_f1, val_acc, val_mse, val_loss, _, _ = self.evaluate(self.data.val_mask)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_macro_f1)
            self.val_micro_f1_scores.append(val_micro_f1)
            self.mse_scores.append(val_mse)
            
            # Test evaluation (for tracking)
            test_micro_f1, test_macro_f1, test_acc, test_mse, test_loss, _, _ = self.evaluate(self.data.test_mask)
            self.test_accuracies.append(test_acc)
            self.test_micro_f1s.append(test_micro_f1)
            
            if val_micro_f1 > best_val_f1:
                best_val_f1 = val_micro_f1
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                # Store early stopping metrics at the best point
                self.early_stop_epoch = epoch
                self.early_stop_train_loss = train_loss
                self.early_stop_val_f1 = val_micro_f1
            else:
                patience_counter += 1
            
            if verbose and epoch % 20 == 0:
                print(f'  Epoch {epoch:03d}: Train_Loss={train_loss:.4f}, Val_Loss={val_loss:.4f}, '
                      f'Val_F1={val_micro_f1:.4f}, Test_Acc={test_acc:.4f}')
            
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                    print(f"  Best epoch: {self.early_stop_epoch}, Train Loss: {self.early_stop_train_loss:.4f}, Val F1: {self.early_stop_val_f1:.4f}")
                break
        
        # Load best model for final evaluation
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        if verbose:
            print(f"Training complete! Trained for {len(self.train_losses)} epochs")
            print(f"Best validation micro-F1: {best_val_f1:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'val_micro_f1_scores': self.val_micro_f1_scores,
            'test_accuracies': self.test_accuracies,
            'test_micro_f1s': self.test_micro_f1s,
            'mse_scores': self.mse_scores,
            'early_stop_epoch': self.early_stop_epoch,
            'early_stop_train_loss': self.early_stop_train_loss,
            'early_stop_val_f1': self.early_stop_val_f1
        }

class AttentionVisualizer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
    def visualize_attention_weights(self, node_idx=0, layer_idx=0, save_path='results/attention_weights.png'):
        """Visualize attention weights for GAT model"""
        if not isinstance(self.model, GAT):
            print("Attention visualization only available for GAT models")
            return
        
        self.model.eval()
        with torch.no_grad():
            # Get attention weights
            _, attention_weights = self.model(self.data.x, self.data.edge_index, return_attention_weights=True)
            
            if layer_idx >= len(attention_weights):
                print(f"Layer {layer_idx} not available. Model has {len(attention_weights)} layers.")
                return
            
            edge_index, alpha = attention_weights[layer_idx]
            
            # Find edges connected to the target node
            node_edges = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
            node_attention = alpha[node_edges].cpu().numpy()
            
            if len(node_attention) == 0:
                print(f"No edges found for node {node_idx}")
                return
            
            # Create attention visualization
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Attention weights distribution
            plt.subplot(2, 2, 1)
            plt.hist(node_attention.flatten(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Attention Weights Distribution\nNode {node_idx}, Layer {layer_idx}')
            plt.xlabel('Attention Weight')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Top attention weights
            plt.subplot(2, 2, 2)
            if len(node_attention) > 0:
                top_indices = np.argsort(node_attention.flatten())[-10:]
                top_weights = node_attention.flatten()[top_indices]
                plt.barh(range(len(top_weights)), top_weights)
                plt.title('Top 10 Attention Weights')
                plt.xlabel('Attention Weight')
                plt.ylabel('Edge Index')
                plt.grid(True, alpha=0.3)
            
            # Plot 3: Attention weights by head (if multi-head)
            plt.subplot(2, 2, 3)
            if node_attention.ndim > 1 and node_attention.shape[1] > 1:
                for head in range(min(4, node_attention.shape[1])):
                    plt.plot(node_attention[:, head], label=f'Head {head}', alpha=0.7)
                plt.title('Attention Weights by Head')
                plt.xlabel('Edge Index')
                plt.ylabel('Attention Weight')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.plot(node_attention.flatten(), 'b-', alpha=0.7)
                plt.title('Attention Weights')
                plt.xlabel('Edge Index')
                plt.ylabel('Attention Weight')
                plt.grid(True, alpha=0.3)
            
            # Plot 4: Summary statistics
            plt.subplot(2, 2, 4)
            stats_text = f"""
            Attention Statistics for Node {node_idx}:
            
            Mean: {np.mean(node_attention):.4f}
            Std:  {np.std(node_attention):.4f}
            Min:  {np.min(node_attention):.4f}
            Max:  {np.max(node_attention):.4f}
            
            Total Edges: {len(node_attention)}
            Layer: {layer_idx}
            Heads: {node_attention.shape[1] if node_attention.ndim > 1 else 1}
            """
            plt.text(0.1, 0.5, stats_text, fontsize=10, transform=plt.gca().transAxes,
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
            plt.axis('off')
            
            plt.tight_layout()
            os.makedirs('results', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Attention visualization saved: {save_path}")

class ExperimentRunner:
    def __init__(self, dataset_name="Cora"):
        self.results = {}
        self.training_curves = {}
        self.early_stop_summary = {}
        
        # Load Cora dataset
        data_loader = CoraDataLoader(dataset_name)
        self.data, self.num_classes = data_loader.load_data()
        
        # Best configurations from ablation study
        self.best_configs = {
            'GCN': {
                'num_layers': 2, 
                'hidden_channels': 64, 
                'dropout': 0.5, 
                'lr': 0.01, 
                'weight_decay': 0.0005
            },
            'GAT': {
                'num_layers': 2, 
                'hidden_channels': 64, 
                'heads': 4, 
                'dropout': 0.5, 
                'attn_dropout': 0.0, 
                'alpha': 0.2, 
                'lr': 0.01, 
                'weight_decay': 0.0005
            },
            'GraphSAGE': {
                'num_layers': 2, 
                'hidden_channels': 128, 
                'dropout': 0.5, 
                'aggregator': 'mean', 
                'lr': 0.01, 
                'weight_decay': 0.0005
            },
            'GraphTransformer': {
                'gcn_layers': 1, 
                'transformer_layers': 2, 
                'hidden_channels': 256, 
                'dropout': 0.5, 
                'attn_dropout': 0.0, 
                'pre_norm': True, 
                'residual_dropout': 0.0, 
                'grad_clip_norm': 1.0, 
                'lr': 0.005, 
                'weight_decay': 0.0005
            }
        }
    
    def create_model(self, model_type, config):
        """Create model based on type and configuration"""
        in_channels = self.data.x.size(1)
        
        if model_type == 'GCN':
            return GCN(
                in_channels=in_channels,
                hidden_channels=config['hidden_channels'],
                out_channels=self.num_classes,
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
        elif model_type == 'GAT':
            return GAT(
                in_channels=in_channels,
                hidden_channels=config['hidden_channels'],
                out_channels=self.num_classes,
                heads=config['heads'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                attn_dropout=config.get('attn_dropout', 0.0),
                alpha=config.get('alpha', 0.2)
            )
        elif model_type == 'GraphSAGE':
            return GraphSAGE(
                in_channels=in_channels,
                hidden_channels=config['hidden_channels'],
                out_channels=self.num_classes,
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                aggregator=config.get('aggregator', 'mean')
            )
        elif model_type == 'GraphTransformer':
            return GraphTransformer(
                in_channels=in_channels,
                hidden_channels=config['hidden_channels'],
                out_channels=self.num_classes,
                gcn_layers=config['gcn_layers'],
                transformer_layers=config['transformer_layers'],
                dropout=config['dropout'],
                attn_dropout=config.get('attn_dropout', 0.0),
                pre_norm=config.get('pre_norm', True),
                residual_dropout=config.get('residual_dropout', 0.0)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_experiments(self):
        """Run experiments with best configurations"""
        print("Starting GNN Experiments with Best Configurations")
        print("="*60)
        
        models = ['GCN', 'GAT', 'GraphSAGE', 'GraphTransformer']
        
        for model_type in models:
            print(f"\n{'='*20} {model_type} EXPERIMENT {'='*20}")
            
            config = self.best_configs[model_type]
            print(f"Using configuration: {config}")
            
            # Create model
            model = self.create_model(model_type, config)
            
            # Create trainer
            trainer = AdvancedTrainer(
                model, 
                self.data, 
                lr=config['lr'],
                weight_decay=config.get('weight_decay', 5e-4),
                grad_clip_norm=config.get('grad_clip_norm', None)
            )
            
            # Train model
            print(f"Training {model_type}...")
            training_results = trainer.train(epochs=200, patience=50, verbose=True)
            
            # Final evaluation
            test_micro_f1, test_macro_f1, test_acc, test_mse, _, y_true, y_pred = trainer.evaluate(self.data.test_mask)
            val_micro_f1, val_macro_f1, val_acc, val_mse, _, _, _ = trainer.evaluate(self.data.val_mask)
            
            # Store early stopping info
            self.early_stop_summary[model_type] = {
                'early_stop_epoch': training_results['early_stop_epoch'],
                'early_stop_train_loss': training_results['early_stop_train_loss'],
                'early_stop_val_f1': training_results['early_stop_val_f1']
            }
            
            # Store results
            self.results[model_type] = {
                'test_accuracy': test_acc,
                'test_micro_f1': test_micro_f1,
                'test_macro_f1': test_macro_f1,
                'val_accuracy': val_acc,
                'val_micro_f1': val_micro_f1,
                'val_macro_f1': val_macro_f1,
                'mse': test_mse,
                'config': config,
                'training_results': training_results,
                'y_true': y_true,
                'y_pred': y_pred,
                'model': model,
                'early_stop_info': self.early_stop_summary[model_type]
            }
            
            self.training_curves[model_type] = training_results
            
            print(f"\n{model_type} FINAL RESULTS:")
            print("="*40)
            print(f"Test Accuracy:      {test_acc:.4f} ({test_acc*100:.1f}%)")
            print(f"Test Micro-F1:      {test_micro_f1:.4f}")
            print(f"Test Macro-F1:      {test_macro_f1:.4f}")
            print(f"Validation Micro-F1: {val_micro_f1:.4f}")
            print(f"MSE:               {test_mse:.4f}")
            print(f"Early Stop Epoch:   {training_results['early_stop_epoch']}")
            print(f"Early Stop Train Loss: {training_results['early_stop_train_loss']:.4f}")
            print(f"Early Stop Val F1:  {training_results['early_stop_val_f1']:.4f}")
            print("="*40)
            
            # Visualize attention weights for GAT
            if model_type == 'GAT':
                print("\nGenerating attention weight visualizations...")
                visualizer = AttentionVisualizer(model, self.data)
                visualizer.visualize_attention_weights(
                    node_idx=0, 
                    layer_idx=0, 
                    save_path=f'results/{model_type}_attention_layer0.png'
                )
        
        # Print Early Stopping Summary
        print(f"\n{'='*60}")
        print("EARLY STOPPING SUMMARY FOR ALL MODELS")
        print("="*60)
        print(f"{'Model':<15} {'Stop Epoch':<12} {'Train Loss':<12} {'Val F1':<10}")
        print("-" * 60)
        for model_type, info in self.early_stop_summary.items():
            print(f"{model_type:<15} {info['early_stop_epoch']:<12} {info['early_stop_train_loss']:<12.4f} {info['early_stop_val_f1']:<10.4f}")
        
        return self.results
    
    def create_early_stop_visualization(self):
        """Create visualization showing early stopping points"""
        plt.figure(figsize=(14, 10))
        
        models = list(self.results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Plot 1: Training Loss with Early Stop Points
        plt.subplot(2, 2, 1)
        for i, model in enumerate(models):
            train_losses = self.training_curves[model]['train_losses']
            epochs = list(range(len(train_losses)))
            plt.plot(epochs, train_losses, label=model, color=colors[i], linewidth=2, alpha=0.8)
            
            # Mark early stopping point
            early_stop_epoch = self.results[model]['early_stop_info']['early_stop_epoch']
            early_stop_loss = self.results[model]['early_stop_info']['early_stop_train_loss']
            plt.scatter(early_stop_epoch, early_stop_loss, color=colors[i], s=100, marker='X', 
                       edgecolors='black', linewidth=2, zorder=5)
        
        plt.title('Training Loss with Early Stopping Points', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Validation F1 with Early Stop Points
        plt.subplot(2, 2, 2)
        for i, model in enumerate(models):
            val_f1s = self.training_curves[model]['val_micro_f1_scores']
            epochs = list(range(len(val_f1s)))
            plt.plot(epochs, val_f1s, label=model, color=colors[i], linewidth=2, alpha=0.8)
            
            # Mark early stopping point
            early_stop_epoch = self.results[model]['early_stop_info']['early_stop_epoch']
            early_stop_val_f1 = self.results[model]['early_stop_info']['early_stop_val_f1']
            plt.scatter(early_stop_epoch, early_stop_val_f1, color=colors[i], s=100, marker='X', 
                       edgecolors='black', linewidth=2, zorder=5)
        
        plt.title('Validation F1 with Early Stopping Points', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Micro-F1', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Early Stop Epochs Comparison
        plt.subplot(2, 2, 3)
        early_stop_epochs = [self.results[model]['early_stop_info']['early_stop_epoch'] for model in models]
        bars = plt.bar(models, early_stop_epochs, color=colors, alpha=0.8, edgecolor='black')
        plt.title('Early Stopping Epochs', fontsize=14, fontweight='bold')
        plt.ylabel('Epoch', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, epoch in zip(bars, early_stop_epochs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{epoch}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Early Stop Values Table
        plt.subplot(2, 2, 4)
        table_data = []
        for model in models:
            info = self.results[model]['early_stop_info']
            table_data.append([
                model,
                f"{info['early_stop_epoch']}",
                f"{info['early_stop_train_loss']:.4f}",
                f"{info['early_stop_val_f1']:.4f}"
            ])
        
        table = plt.table(cellText=table_data,
                         colLabels=['Model', 'Stop Epoch', 'Train Loss', 'Val F1'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Color the table
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#2C3E50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data) + 1):
            model_color = colors[(i-1) % len(colors)]
            for j in range(len(table_data[0])):
                if j == 0:
                    table[(i, j)].set_facecolor(model_color)
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    import matplotlib.colors as mcolors
                    light_color = mcolors.to_rgba(model_color, alpha=0.3)
                    table[(i, j)].set_facecolor(light_color)
                    table[(i, j)].set_text_props(weight='bold')
        
        plt.axis('off')
        plt.title('Early Stopping Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/early_stopping_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: results/early_stopping_analysis.png")

    def create_summary_table(self):
        """Create a nicely formatted summary table"""
        plt.figure(figsize=(14, 6))
        
        # Prepare data for table
        table_data = []
        for model_name, results in self.results.items():
            table_data.append([
                model_name,
                f"{results['test_accuracy']:.4f}",
                f"{results['test_micro_f1']:.4f}",
                f"{results['test_macro_f1']:.4f}",
                f"{results['val_micro_f1']:.4f}",
                f"{results['mse']:.4f}"
            ])
        
        # Create table
        table = plt.table(cellText=table_data,
                         colLabels=['Model', 'Test Accuracy', 'Test Micro-F1', 
                                   'Test Macro-F1', 'Val Micro-F1', 'MSE'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2.5)
        
        # Color header
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#2C3E50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by model
        for i in range(1, len(table_data) + 1):
            model_color = colors[(i-1) % len(colors)]
            for j in range(len(table_data[0])):
                if j == 0:  # Model name column
                    table[(i, j)].set_facecolor(model_color)
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    # Lighter shade for data columns
                    import matplotlib.colors as mcolors
                    light_color = mcolors.to_rgba(model_color, alpha=0.3)
                    table[(i, j)].set_facecolor(light_color)
                    table[(i, j)].set_text_props(weight='bold')
        
        plt.axis('off')
        plt.title('GNN Models Performance Summary', 
                 fontsize=18, fontweight='bold', pad=30)
        plt.tight_layout()
        
        plt.savefig('results/summary_table.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: results/summary_table.png")

    def create_visualizations(self):
        """Create individual visualization files"""
        print("\nCreating individual visualizations...")
        
        os.makedirs('results', exist_ok=True)
        
        models = list(self.results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 1. Training Loss vs Epoch
        plt.figure(figsize=(10, 6))
        for i, model in enumerate(models):
            train_losses = self.training_curves[model]['train_losses']
            plt.plot(train_losses, label=model, color=colors[i], linewidth=3, alpha=0.8)
        plt.title('Training Loss vs Epoch', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Training Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/training_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: results/training_loss.png")
        
        # 2. Validation Loss vs Epoch
        plt.figure(figsize=(10, 6))
        for i, model in enumerate(models):
            val_losses = self.training_curves[model]['val_losses']
            plt.plot(val_losses, label=model, color=colors[i], linewidth=3, alpha=0.8)
        plt.title('Validation Loss vs Epoch', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/validation_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: results/validation_loss.png")
        
        # 3. Test Accuracy Comparison
        plt.figure(figsize=(10, 6))
        test_accs = [self.results[model]['test_accuracy'] for model in models]
        bars = plt.bar(models, test_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Test Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Models', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, acc in zip(bars, test_accs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, max(test_accs) * 1.1)
        plt.tight_layout()
        plt.savefig('results/test_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: results/test_accuracy.png")
        
        # 4. Test Micro-F1 Comparison
        plt.figure(figsize=(10, 6))
        test_f1s = [self.results[model]['test_micro_f1'] for model in models]
        bars = plt.bar(models, test_f1s, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Test Micro-F1 Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Test Micro-F1 Score', fontsize=14, fontweight='bold')
        plt.xlabel('Models', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, f1 in zip(bars, test_f1s):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, max(test_f1s) * 1.1)
        plt.tight_layout()
        plt.savefig('results/test_micro_f1.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: results/test_micro_f1.png")
        
        # 5. Validation Micro-F1 vs Epoch
        plt.figure(figsize=(10, 6))
        for i, model in enumerate(models):
            val_f1s = self.training_curves[model]['val_micro_f1_scores']
            plt.plot(val_f1s, label=model, color=colors[i], linewidth=3, alpha=0.8)
        plt.title('Validation Micro-F1 vs Epoch', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Validation Micro-F1 Score', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/validation_micro_f1.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: results/validation_micro_f1.png")
        
        # 6. MSE Comparison
        plt.figure(figsize=(10, 6))
        mse_scores = [self.results[model]['mse'] for model in models]
        bars = plt.bar(models, mse_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Mean Squared Error (MSE) Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('MSE Score', fontsize=14, fontweight='bold')
        plt.xlabel('Models', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, mse in zip(bars, mse_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{mse:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, max(mse_scores) * 1.15)
        plt.tight_layout()
        plt.savefig('results/mse_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: results/mse_comparison.png")
        
        # 7. Test Macro-F1 Comparison (Bonus)
        plt.figure(figsize=(10, 6))
        test_macro_f1s = [self.results[model]['test_macro_f1'] for model in models]
        bars = plt.bar(models, test_macro_f1s, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Test Macro-F1 Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Test Macro-F1 Score', fontsize=14, fontweight='bold')
        plt.xlabel('Models', fontsize=14, fontweight='bold')
        
        for bar, f1 in zip(bars, test_macro_f1s):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, max(test_macro_f1s) * 1.1)
        plt.tight_layout()
        plt.savefig('results/test_macro_f1.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: results/test_macro_f1.png")
        
        # 8. Summary Table
        self.create_summary_table()
        
        # 9. Early Stopping Analysis
        self.create_early_stop_visualization()

    def save_results(self):
        """Save all results to CSV files"""
        print("\nSaving results...")
        
        os.makedirs('results', exist_ok=True)
        
        # 1. Main results table with early stopping info
        main_results = []
        for model_name, results in self.results.items():
            main_results.append({
                'Model': model_name,
                'Test_Accuracy': f"{results['test_accuracy']:.4f}",
                'Test_Micro_F1': f"{results['test_micro_f1']:.4f}",
                'Test_Macro_F1': f"{results['test_macro_f1']:.4f}",
                'Val_Micro_F1': f"{results['val_micro_f1']:.4f}",
                'MSE': f"{results['mse']:.4f}",
                'Early_Stop_Epoch': results['early_stop_info']['early_stop_epoch'],
                'Early_Stop_Train_Loss': f"{results['early_stop_info']['early_stop_train_loss']:.4f}",
                'Early_Stop_Val_F1': f"{results['early_stop_info']['early_stop_val_f1']:.4f}",
                'Best_Config': str(results['config'])
            })
        
        pd.DataFrame(main_results).to_csv('results/main_results.csv', index=False)
        print("Saved: results/main_results.csv")
        
        # 2. Early stopping summary
        early_stop_data = []
        for model_name, info in self.early_stop_summary.items():
            early_stop_data.append({
                'Model': model_name,
                'Early_Stop_Epoch': info['early_stop_epoch'],
                'Early_Stop_Train_Loss': f"{info['early_stop_train_loss']:.4f}",
                'Early_Stop_Val_F1': f"{info['early_stop_val_f1']:.4f}"
            })
        
        pd.DataFrame(early_stop_data).to_csv('results/early_stopping_summary.csv', index=False)
        print("Saved: results/early_stopping_summary.csv")
        
        # 3. Training curves data
        max_epochs = max(len(curves['train_losses']) for curves in self.training_curves.values())
        
        curves_data = {'Epoch': list(range(max_epochs))}
        for model_name, curves in self.training_curves.items():
            # Pad shorter lists with NaN
            for metric_name, values in curves.items():
                if isinstance(values, list):  # Only for list metrics, not single values
                    padded_values = values + [np.nan] * (max_epochs - len(values))
                    curves_data[f'{model_name}_{metric_name}'] = padded_values
        
        pd.DataFrame(curves_data).to_csv('results/training_curves.csv', index=False)
        print("Saved: results/training_curves.csv")
        
        # 4. Best configurations
        config_data = []
        for model_name, config in self.best_configs.items():
            config_data.append({
                'Model': model_name,
                **config
            })
        
        pd.DataFrame(config_data).to_csv('results/best_configurations.csv', index=False)
        print("Saved: results/best_configurations.csv")

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*70)
        print("GNN EXPERIMENT RESULTS SUMMARY")
        print("="*70)
        
        print("\nEARLY STOPPING SUMMARY:")
        print("-" * 50)
        print(f"{'Model':<15} {'Stop Epoch':<12} {'Train Loss':<12} {'Val F1':<10}")
        print("-" * 50)
        for model_name, info in self.early_stop_summary.items():
            print(f"{model_name:<15} {info['early_stop_epoch']:<12} {info['early_stop_train_loss']:<12.4f} {info['early_stop_val_f1']:<10.4f}")
        
        print("\nFINAL RESULTS RANKING (by Test Accuracy):")
        print("-" * 50)
        
        # Sort models by test accuracy
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_accuracy'], 
                             reverse=True)
        
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            print(f"{rank}. {model_name:15s} - Acc: {results['test_accuracy']:.4f} "
                  f"| F1: {results['test_micro_f1']:.4f} | MSE: {results['mse']:.4f}")
        
        print("\nCONFIGURATIONS USED:")
        print("-" * 50)
        for model_name, config in self.best_configs.items():
            print(f"{model_name:15s}: {config}")
        
        print(f"\nAll results saved in 'results/' directory")
        print("Individual PNG files created:")
        print("  1. training_loss.png")
        print("  2. validation_loss.png") 
        print("  3. test_accuracy.png")
        print("  4. test_micro_f1.png")
        print("  5. validation_micro_f1.png")
        print("  6. mse_comparison.png")
        print("  7. test_macro_f1.png")
        print("  8. summary_table.png")
        print("  9. early_stopping_analysis.png")
        print("  10. GAT_attention_layer0.png")
        print("\nCSV files:")
        print("  - main_results.csv (includes early stopping info)")
        print("  - early_stopping_summary.csv")
        print("  - training_curves.csv")
        print("  - best_configurations.csv")
        print("="*70)
    
    def run_all_experiments(self):
        """Run all experiments"""
        print("STARTING OPTIMIZED GNN EXPERIMENTS")
        print("="*60)
        
        # Run experiments with best configurations
        results = self.run_experiments()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return results

if __name__ == "__main__":
    # Run experiments with best configurations
    runner = ExperimentRunner(dataset_name="Cora")
    results = runner.run_all_experiments()