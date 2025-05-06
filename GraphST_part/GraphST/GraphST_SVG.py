import torch
from .GraphST import GraphST
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from .preprocess import permutation

def calculate_spatial_variability(X, coordinates, n_neighbors=6):
    """
    计算基因的空间变异性得分
    
    参数:
    X: numpy array, 基因表达矩阵 (spots × genes)
    coordinates: numpy array, 空间坐标 (spots × 2)
    n_neighbors: int, 用于计算局部变异性的邻居数量
    
    返回:
    spatial_scores: 每个基因的空间变异性得分
    """
    from sklearn.neighbors import NearestNeighbors
    
    # 找到每个spot的最近邻
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)
    
    # 计算局部变异性
    local_var = np.zeros(X.shape[1])
    for i in range(X.shape[1]):  # 对每个基因
        gene_expr = X[:, i]
        # 计算每个spot与其邻居的表达差异
        neighbor_var = np.array([
            np.var(gene_expr[indices[j]]) for j in range(len(indices))
        ])
        # 计算整体变异性得分
        local_var[i] = np.mean(neighbor_var)
    
    return local_var

def get_svg_genes_fast(adata, n_top_genes=3000, min_expr=0.1):
    """
    使用简化的方法快速识别空间变异基因
    
    参数:
    adata: AnnData对象
    n_top_genes: 选择的SVG数量
    min_expr: 最小表达量阈值
    
    返回:
    svg_genes: SVG基因列表
    """
    # 准备数据
    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # 基本过滤
    gene_means = np.mean(X, axis=0)
    valid_genes = gene_means > min_expr
    X = X[:, valid_genes]
    valid_gene_names = adata.var_names[valid_genes]
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 计算空间变异性得分
    spatial_scores = calculate_spatial_variability(
        X_scaled, 
        adata.obsm['spatial']
    )
    
    # 选择得分最高的基因
    top_indices = np.argsort(spatial_scores)[-n_top_genes:]
    selected_genes = valid_gene_names[top_indices]
    
    return list(selected_genes)

def get_svg_genes_scanpy(adata, n_top_genes=3000, min_mean=0.0125, max_mean=3, min_disp=0.5):
    """
    使用scanpy的方法识别变异基因，并结合空间信息
    
    参数:
    adata: AnnData对象
    n_top_genes: 选择的基因数量
    min_mean: 最小平均表达量
    max_mean: 最大平均表达量
    min_disp: 最小离散度
    
    返回:
    svg_genes: 变异基因列表
    """
    # 复制adata以避免修改原始数据
    adata_temp = adata.copy()
    
    # 标准预处理
    if 'highly_variable' in adata_temp.var:
        del adata_temp.var['highly_variable']
    
    # 使用scanpy计算高变基因
    sc.pp.highly_variable_genes(
        adata_temp,
        n_top_genes=n_top_genes,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        batch_key=None,
        flavor='seurat_v3'
    )
    
    return adata_temp.var_names[adata_temp.var['highly_variable']].tolist()

class GraphST_SVG(GraphST):
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
        datatype = '10X',
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5
        ):
        """
        GraphST with SVG (Spatially Variable Genes) filtering using scanpy methods
        
        Parameters
        ----------
        与原始GraphST参数相同，另加：
        min_mean : float, optional (default: 0.0125)
            最小平均表达量
        max_mean : float, optional (default: 3)
            最大平均表达量
        min_disp : float, optional (default: 0.5)
            最小离散度
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
        from .preprocess import fix_seed
        fix_seed(self.random_seed)
        
        # 基本预处理
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # 获取变异基因
        print("Identifying variable genes using scanpy...")
        svg_genes = get_svg_genes_scanpy(
            self.adata, 
            n_top_genes=3000,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp
        )
        print(f"Number of identified variable genes: {len(svg_genes)}")
        self.adata.var['highly_variable'] = self.adata.var_names.isin(svg_genes)
        
        # 对选定的基因进行缩放
        sc.pp.scale(self.adata, zero_center=False, max_value=10)
        
        # 继续初始化
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
           
        # 获取特征
        if 'feat' not in self.adata.obsm.keys():
            if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
                feat = self.adata[:, self.adata.var['highly_variable']].X.toarray()[:, ]
            else:
                feat = self.adata[:, self.adata.var['highly_variable']].X[:, ]
            
            # 数据增强
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