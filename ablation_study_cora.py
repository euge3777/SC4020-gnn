import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load Cora dataset
def load_cora_data():
    transform = T.NormalizeFeatures()
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=transform)
    data = dataset[0]
    print(f"Cora Dataset - Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}")
    return data, dataset.num_classes

# Model Definitions
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2, 
                 dropout=0.5, attn_dropout=0.0, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                                 dropout=attn_dropout, negative_slope=alpha))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, 
                                     dropout=attn_dropout, negative_slope=alpha))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, 
                                 dropout=attn_dropout, negative_slope=alpha))
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.5, aggregator='mean'):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        
        # Note: PyTorch Geometric SAGEConv uses 'mean' aggregation by default
        # For other aggregators, you might need custom implementation
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
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

# Training and Evaluation
class FastTrainer:
    def __init__(self, model, data, lr=0.01, weight_decay=5e-4, grad_clip_norm=None):
        self.model = model.to(DEVICE)
        self.data = data.to(DEVICE)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip_norm = grad_clip_norm
        
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
            
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='micro')
            
            return accuracy, f1
    
    def train_and_evaluate(self, epochs=100, patience=20):
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            loss = self.train_epoch()
            val_acc, val_f1 = self.evaluate(self.data.val_mask)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model state
                torch.save(self.model.state_dict(), 'temp_best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Load best model and get final test results
        self.model.load_state_dict(torch.load('temp_best_model.pth'))
        test_acc, test_f1 = self.evaluate(self.data.test_mask)
        val_acc, val_f1 = self.evaluate(self.data.val_mask)
        
        return {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'epochs_trained': epoch + 1
        }

# Ablation Study Runner
class AblationStudyRunner:
    def __init__(self, data, num_classes):
        self.data = data
        self.num_classes = num_classes
        self.results = {}
        
        # Define optimized hyperparameter grids for each model
        self.param_grids = {
            "GCN": {
                # FAST
                "num_layers": [2, 3, 4],
                "hidden_channels": [32, 64, 128, 256],
                "dropout": [0, 0.3, 0.5],
                "lr": [0.005, 0.01],
                "weight_decay": [5e-4, 1e-3],
                # EXTENDED (uncomment to widen search)
                # "num_layers": [2, 3, 4],
                # "hidden_channels": [16, 32, 64, 128],
                # "dropout": [0.0, 0.3, 0.5, 0.7],
                # "lr": [0.003, 0.005, 0.01],
                # "weight_decay": [1e-4, 5e-4, 1e-3]
            },

            "GAT": {
                # FAST
                "num_layers": [2, 3, 4],                 # deeper tends to hurt on Cora
                "hidden_channels": [32, 64],
                "heads": [4],                      # 4–8 typical; 4 is a good trade-off
                "dropout": [0, 0.5, 0.7],             # higher dropout stabilizes GAT
                "attn_dropout": [0.0, 0.2],        # add a separate attention dropout
                "alpha": [0.2],                    # LeakyReLU negative slope
                "lr": [0.005, 0.01],
                "weight_decay": [5e-4, 1e-3],
                # EXTENDED
                # "num_layers": [2, 3],
                # "heads": [2, 4, 8],
                # "dropout": [0.3, 0.5, 0.7],
                # "attn_dropout": [0.0, 0.2, 0.4],
                # "alpha": [0.1, 0.2, 0.3],
                # "lr": [0.003, 0.005, 0.01],
                # "weight_decay": [1e-4, 5e-4, 1e-3]
            },

            "GraphSAGE": {
                # FAST
                "num_layers": [2, 3, 4],
                "hidden_channels": [32, 64, 128],
                "dropout": [0, 0.3, 0.5],             # 0.7 often too strong on Cora
                "lr": [0.005, 0.01],
                "weight_decay": [5e-4, 1e-3],
                # If your implementation supports it:
                "aggregator": ["mean"],            # EXTENDED: ["mean", "max", "lstm"]
                # EXTENDED
                # "num_layers": [2, 3],
                # "hidden_channels": [32, 64, 128],
                # "dropout": [0.0, 0.3, 0.5],
                # "lr": [0.003, 0.005, 0.01],
                # "weight_decay": [1e-4, 5e-4, 1e-3],
                # "aggregator": ["mean", "max", "lstm"]
            },

            "GraphTransformer": {
                # Using your decomposition (gcn_layers + transformer_layers)
                # FAST
                "gcn_layers": [0, 1, 2, 3, 4],              # keep locality light; too much → smoothing
                "transformer_layers": [1, 2, 3],      # 2–3 worked best in your runs
                "hidden_channels": [64, 128, 256],           # transformers are capacity-heavy
                "dropout": [0, 0.5, 0.6],             # a touch higher to tame overfit on Cora
                "attn_dropout": [0.0, 0.2],
                "lr": [0.003, 0.005],              # slightly LOWER than 0.01 to reduce bumps
                "weight_decay": [5e-4, 1e-3],
                # Optional stabilizers (set if your model supports them)
                "pre_norm": [True],                # LayerNorm before blocks improves stability
                "residual_dropout": [0.0, 0.1],
                "grad_clip_norm": [1.0],           # implement outside grid if easier
                # EXTENDED
                # "gcn_layers": [0, 1, 2],
                # "transformer_layers": [1, 2, 3],
                # "hidden_channels": [32, 64, 128],
                # "dropout": [0.3, 0.5, 0.6],
                # "attn_dropout": [0.0, 0.2, 0.4],
                # "lr": [0.002, 0.003, 0.005],
                # "weight_decay": [1e-4, 5e-4, 1e-3],
                # "pre_norm": [True, False],
                # "residual_dropout": [0.0, 0.1, 0.2]
            }
        }
        
        # Base configurations
        self.base_configs = {
            'GCN': {'num_layers': 2, 'hidden_channels': 64, 'dropout': 0.5, 'lr': 0.01, 'weight_decay': 5e-4},
            'GAT': {'num_layers': 2, 'hidden_channels': 64, 'heads': 4, 'dropout': 0.5, 'attn_dropout': 0.0, 'alpha': 0.2, 'lr': 0.01, 'weight_decay': 5e-4},
            'GraphSAGE': {'num_layers': 2, 'hidden_channels': 64, 'dropout': 0.5, 'aggregator': 'mean', 'lr': 0.01, 'weight_decay': 5e-4},
            'GraphTransformer': {'gcn_layers': 1, 'transformer_layers': 2, 'hidden_channels': 64, 'dropout': 0.5, 'attn_dropout': 0.0, 'pre_norm': True, 'residual_dropout': 0.0, 'grad_clip_norm': 1.0, 'lr': 0.005, 'weight_decay': 5e-4}
        }
    
    def create_model(self, model_type, config):
        """Create model instance based on type and configuration"""
        in_channels = self.data.x.size(1)
        
        if model_type == 'GCN':
            return GCN(in_channels, config['hidden_channels'], self.num_classes,
                      num_layers=config['num_layers'], dropout=config['dropout'])
        elif model_type == 'GAT':
            return GAT(in_channels, config['hidden_channels'], self.num_classes,
                      heads=config['heads'], num_layers=config['num_layers'], 
                      dropout=config['dropout'], attn_dropout=config.get('attn_dropout', 0.0),
                      alpha=config.get('alpha', 0.2))
        elif model_type == 'GraphSAGE':
            return GraphSAGE(in_channels, config['hidden_channels'], self.num_classes,
                            num_layers=config['num_layers'], dropout=config['dropout'],
                            aggregator=config.get('aggregator', 'mean'))
        elif model_type == 'GraphTransformer':
            return GraphTransformer(in_channels, config['hidden_channels'], self.num_classes,
                                   gcn_layers=config['gcn_layers'], 
                                   transformer_layers=config['transformer_layers'],
                                   dropout=config['dropout'],
                                   attn_dropout=config.get('attn_dropout', 0.0),
                                   pre_norm=config.get('pre_norm', True),
                                   residual_dropout=config.get('residual_dropout', 0.0))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_single_experiment(self, model_type, config):
        """Run a single experiment with given configuration"""
        model = self.create_model(model_type, config)
        trainer = FastTrainer(model, self.data, lr=config['lr'], 
                             weight_decay=config.get('weight_decay', 5e-4),
                             grad_clip_norm=config.get('grad_clip_norm', None))
        results = trainer.train_and_evaluate(epochs=100)
        return results
    
    def run_ablation_for_model(self, model_type):
        """Run ablation study for a specific model"""
        print(f"\n{'='*20} {model_type} ABLATION STUDY {'='*20}")
        
        base_config = self.base_configs[model_type].copy()
        param_grid = self.param_grids[model_type]
        
        all_results = []
        best_config = base_config.copy()
        best_score = 0
        
        # Test each parameter individually
        for param_name, param_values in param_grid.items():
            print(f"\nTesting {param_name}: {param_values}")
            
            with tqdm(param_values, desc=f"{param_name}") as pbar:
                for value in pbar:
                    # Create config with current parameter value
                    config = base_config.copy()
                    config[param_name] = value
                    
                    # Run experiment
                    try:
                        result = self.run_single_experiment(model_type, config)
                        
                        # Store result
                        result_record = {
                            'model': model_type,
                            'parameter': param_name,
                            'value': value,
                            'config': config.copy(),
                            **result
                        }
                        all_results.append(result_record)
                        
                        # Update best config if this is better
                        if result['val_f1'] > best_score:
                            best_score = result['val_f1']
                            best_config[param_name] = value
                        
                        pbar.set_postfix({
                            'Val_F1': f"{result['val_f1']:.3f}",
                            'Test_Acc': f"{result['test_accuracy']:.3f}"
                        })
                        
                    except Exception as e:
                        print(f"Error with {param_name}={value}: {e}")
                        continue
        
        # Test best configuration
        print(f"\nTesting best configuration: {best_config}")
        final_result = self.run_single_experiment(model_type, best_config)
        
        print(f"Best {model_type} Configuration:")
        print(f"  Config: {best_config}")
        print(f"  Test Accuracy: {final_result['test_accuracy']:.4f}")
        print(f"  Test F1: {final_result['test_f1']:.4f}")
        print(f"  Validation F1: {final_result['val_f1']:.4f}")
        
        self.results[model_type] = {
            'best_config': best_config,
            'best_results': final_result,
            'all_results': all_results
        }
        
        return best_config, final_result, all_results
    
    def run_all_ablations(self):
        """Run ablation studies for all models"""
        print("STARTING OPTIMIZED ABLATION STUDY")
        print("="*60)
        
        models = ['GCN', 'GAT', 'GraphSAGE', 'GraphTransformer']
        
        for model_type in models:
            self.run_ablation_for_model(model_type)
        
        return self.results
    
    def save_results(self):
        """Save ablation study results"""
        os.makedirs('ablation_results', exist_ok=True)
        
        # Save detailed results for each model
        for model_type, model_results in self.results.items():
            # Save all experimental results
            df = pd.DataFrame(model_results['all_results'])
            df.to_csv(f'ablation_results/{model_type}_ablation_detailed.csv', index=False)
            
            # Save best configuration
            best_config_df = pd.DataFrame([{
                'Model': model_type,
                **model_results['best_config'],
                **model_results['best_results']
            }])
            best_config_df.to_csv(f'ablation_results/{model_type}_best_config.csv', index=False)
        
        # Save summary of all best configurations
        summary_data = []
        for model_type, model_results in self.results.items():
            summary_data.append({
                'Model': model_type,
                'Best_Config': str(model_results['best_config']),
                'Test_Accuracy': model_results['best_results']['test_accuracy'],
                'Test_F1': model_results['best_results']['test_f1'],
                'Val_F1': model_results['best_results']['val_f1']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('ablation_results/ablation_summary.csv', index=False)
        
        print("\nResults saved in 'ablation_results/' directory:")
        print("  - [MODEL]_ablation_detailed.csv (detailed results)")
        print("  - [MODEL]_best_config.csv (best configuration)")
        print("  - ablation_summary.csv (summary of all models)")
    
    def create_visualizations(self):
        """Create visualization plots for ablation study results"""
        os.makedirs('ablation_results', exist_ok=True)
        
        for model_type, model_results in self.results.items():
            self.plot_ablation_results(model_type, model_results['all_results'])
        
        # Create comparison plot
        self.plot_model_comparison()
    
    def plot_ablation_results(self, model_type, results):
        """Plot ablation results for a specific model"""
        df = pd.DataFrame(results)
        
        # Get unique parameters
        parameters = df['parameter'].unique()
        n_params = len(parameters)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_type} Optimized Ablation Study Results', fontsize=16, fontweight='bold')
        
        for i, param in enumerate(parameters):
            if i >= 6:  # Max 6 subplots
                break
                
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            param_data = df[df['parameter'] == param]
            
            # Plot validation F1 vs parameter value
            values = param_data['value'].tolist()
            val_f1s = param_data['val_f1'].tolist()
            
            ax.plot(values, val_f1s, 'o-', linewidth=2, markersize=8)
            ax.set_title(f'{param.replace("_", " ").title()} vs Validation F1')
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Validation F1')
            ax.grid(True, alpha=0.3)
            
            # Highlight best value
            best_idx = np.argmax(val_f1s)
            ax.plot(values[best_idx], val_f1s[best_idx], 'r*', markersize=15, label='Best')
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_params, 6):
            row, col = i // 3, i % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'ablation_results/{model_type}_optimized_ablation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: ablation_results/{model_type}_optimized_ablation_plots.png")
    
    def plot_model_comparison(self):
        """Plot comparison of best configurations across models"""
        plt.figure(figsize=(12, 8))
        
        models = list(self.results.keys())
        test_accs = [self.results[model]['best_results']['test_accuracy'] for model in models]
        test_f1s = [self.results[model]['best_results']['test_f1'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.subplot(1, 2, 1)
        bars1 = plt.bar(x, test_accs, width, label='Test Accuracy', alpha=0.8)
        plt.title('Best Test Accuracy by Model (Optimized)')
        plt.ylabel('Test Accuracy')
        plt.xlabel('Models')
        plt.xticks(x, models, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, acc in zip(bars1, test_accs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 2, 2)
        bars2 = plt.bar(x, test_f1s, width, label='Test F1', alpha=0.8, color='orange')
        plt.title('Best Test F1 by Model (Optimized)')
        plt.ylabel('Test F1')
        plt.xlabel('Models')
        plt.xticks(x, models, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, f1 in zip(bars2, test_f1s):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ablation_results/optimized_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: ablation_results/optimized_model_comparison.png")
    
    def print_summary(self):
        """Print summary of ablation study results"""
        print("\n" + "="*80)
        print("OPTIMIZED ABLATION STUDY SUMMARY")
        print("="*80)
        
        # Sort models by test accuracy
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['best_results']['test_accuracy'], 
                             reverse=True)
        
        print("\nFINAL RANKING (by Test Accuracy):")
        print("-" * 50)
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            best_results = results['best_results']
            print(f"{rank}. {model_name:15s} - Acc: {best_results['test_accuracy']:.4f} | "
                  f"F1: {best_results['test_f1']:.4f}")
        
        print("\nBEST CONFIGURATIONS:")
        print("-" * 50)
        for model_name, results in self.results.items():
            print(f"{model_name:15s}: {results['best_config']}")
        
        print("\nFiles saved in 'ablation_results/' directory")
        print("="*80)

def main():
    """Main function to run optimized ablation study"""
    # Load data
    data, num_classes = load_cora_data()
    
    # Run ablation study
    runner = AblationStudyRunner(data, num_classes)
    results = runner.run_all_ablations()
    
    # Save results
    runner.save_results()
    
    # Create visualizations
    runner.create_visualizations()
    
    # Print summary
    runner.print_summary()
    
    # Clean up temporary files
    if os.path.exists('temp_best_model.pth'):
        os.remove('temp_best_model.pth')

if __name__ == "__main__":
    main()