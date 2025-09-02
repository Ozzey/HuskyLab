import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

class GNNSceneEmbeddingNetwork_LearnedEdgeVector(nn.Module):
    def __init__(self, 
                 object_feature_dim=518, 
                 num_relations=26,  # Number of unique relations
                 node_embedding_dim=128, 
                 edge_embedding_dim=32,
                 scene_embedding_dim=32):
        """
        GNNSceneEmbeddingNetwork_LearnedEdgeVector processes a scene graph with node features and
        categorical edge features to generate a single scene embedding.

        Args:
        object_feature_dim (int): Dimensionality of input features for each object.
        num_relations (int): The total number of unique relation types.
        node_embedding_dim (int): Dimensionality of the intermediate node embeddings.
        edge_embedding_dim (int): Dimensionality of the learned edge embeddings.
        scene_embedding_dim (int): Dimensionality of the final scene embedding.
        """
        super(GNNSceneEmbeddingNetwork_LearnedEdgeVector, self).__init__()

        # 1. Node Feature Encoder: Maps high-dim raw features to a lower-dim space
        self.node_encoder = nn.Sequential(
            nn.Linear(object_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, node_embedding_dim)
        )
        
        # 2. Edge Feature Embedding: Creates a dense vector for each relation type
        # The layer learns a unique vector representation for each categorical relation.
        self.rel_embedding = nn.Embedding(num_relations, edge_embedding_dim)

        # 3. GNN Layers: Use GATConv which can incorporate edge features.
        # It updates each node's representation by attending to its neighbors' features
        # and their relations.
        self.conv1 = GATConv(in_channels=node_embedding_dim,
                             out_channels=node_embedding_dim,
                             edge_dim=edge_embedding_dim) # GATConv supports edge features directly
        
        self.conv2 = GATConv(in_channels=node_embedding_dim,
                             out_channels=node_embedding_dim,
                             edge_dim=edge_embedding_dim)

        # 4. Global Readout Network: Aggregates all node embeddings into a single vector
        # This is a key step to produce a fixed-size representation of the entire graph.
        self.readout_net = nn.Sequential(
            nn.Linear(node_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, scene_embedding_dim)
        )

    def forward(self, data):
        # Data object contains x, edge_index, and edge_attr
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. Encode node features
        x = self.node_encoder(x)  # Shape: (num_nodes, node_embedding_dim)

        # 2. Encode edge features using the embedding layer
        edge_attr = self.rel_embedding(edge_attr)  # Shape: (num_edges, edge_embedding_dim)

        # 3. Message Passing with GNN layers
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        
        # 4. Global pooling to get the scene embedding
        # PyG's global_mean_pool handles pooling over all nodes in the graph
        # This is the "readout" step that summarizes the entire graph.
        x = global_mean_pool(x, data.batch if data.batch is not None else torch.zeros(x.size(0), dtype=torch.long))

        # 5. Final scene-level encoding
        scene_embedding = self.readout_net(x)
        
        return scene_embedding