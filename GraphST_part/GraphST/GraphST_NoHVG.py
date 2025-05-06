import torch
from .GraphST import GraphST
import scanpy as sc
import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd

class GraphST_NoHVG(GraphST):
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X'
        ):
        """
        GraphST without HVG filtering
        """
        # Initialize base class attributes without calling its __init__
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        
        # Set random seed
        from .preprocess import fix_seed, permutation
        fix_seed(self.random_seed)
        
        # Skip HVG filtering, only do normalization
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.scale(self.adata, zero_center=False, max_value=10)
        
        # Continue with the rest of the initialization
        if 'adj' not in self.adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              from .preprocess import construct_interaction_KNN
              construct_interaction_KNN(self.adata)
           else:    
              from .preprocess import construct_interaction
              construct_interaction(self.adata)
         
        if 'label_CSL' not in self.adata.obsm.keys():    
           from .preprocess import add_contrastive_label
           add_contrastive_label(self.adata)
           
        # 直接获取特征，不使用get_feature
        if 'feat' not in self.adata.obsm.keys():
            if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
                feat = self.adata.X.toarray()[:, ]
            else:
                feat = self.adata.X[:, ]
            
            # data augmentation
            feat_a = permutation(feat)
            
            self.adata.obsm['feat'] = feat
            self.adata.obsm['feat_a'] = feat_a
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
    
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           from .preprocess import preprocess_adj_sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           from .preprocess import preprocess_adj
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
           
        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs 