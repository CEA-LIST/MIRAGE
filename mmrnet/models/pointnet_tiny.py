import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, fps, global_max_pool, radius, PointNetConv
from torch_geometric.data import Data

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)
        
    def forward(self, *args):
        # Unpack input arguments
        x = args[0]  # Node features
        pos_conv = args[1]  # Node positions (for convolution)
        batch = args[2]  # Batch index for each node
        pos_graph = args[3] if len(args) == 4 else args[1] # Node positions (for graph construction)

        # Determine sampling indices
        if self.ratio < 1.0:
            # Perform FPS (Farthest Point Sampling) on pos_graph using only x,y,z
            idx = fps(pos_graph[:,:3], batch, ratio=self.ratio)
        else:
            # Use all nodes if no sampling
            idx = torch.arange(len(batch), device=batch.device)

        # Select positions for radius search
        pos_graph_search = pos_graph
        pos_graph_centroids = pos_graph[idx]

        # Perform radius search
        row, col = radius(pos_graph_search, pos_graph_centroids, self.r, batch, batch[idx], max_num_neighbors=64)
        # The `row` are the indices of the centroids (i.e., query points)
        # The `col` are the indices of the neighbors (i.e., points in the search set)

        # Construct edge index
        edge_index = torch.stack([col, row], dim=0)

        # Select positions for convolution
        pos_conv_input = pos_conv
        pos_conv_centroids = pos_conv[idx]

        # Extract features for sampled nodes
        x_dst = x[idx] if x is not None else None

        # Perform convolution
        x = self.conv((x, x_dst), (pos_conv_input, pos_conv_centroids), edge_index)

        # Return the output for the next layer
        if len(args) == 4:
            return x, pos_conv[idx], batch[idx], pos_graph[idx]
        return x, pos_conv[idx], batch[idx]


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn
    
    def forward(self, *args):
        
        x = args[0]
        pos = args[1]
        batch = args[2]

        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet_Tiny(torch.nn.Module):
    def __init__(self, info=None):
        super().__init__()
        self.num_classes = info.get('num_classes')
        
        self.fps_ratios = info.get('fps_ratios') or [0.75, 0.5]
        self.radius = info.get('radius') or [0.2, 0.4]
        self.t_scale = info.get('t_scale') if info.get('t_scale') is not None else None
        print(f"Pointnet using custom arguments: fps_ratios={self.fps_ratios}, radius={self.radius}, t_scale={self.t_scale}")
        
        x_features = info.get('num_features', 3)  # Input feature dimensions
        pos_features = info.get('num_pos_features', 3)  # Position feature dimensions

        self.sa1_module = SAModule(self.fps_ratios[0], self.radius[0], MLP([x_features + pos_features, 32, 64]))
        self.sa2_module = SAModule(self.fps_ratios[1], self.radius[1], MLP([64 + pos_features, 64, 128]))
        self.sa3_module = GlobalSAModule(MLP([128 + x_features, 128, 256]))

        self.mlp = MLP([256, 128, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        if isinstance(data, Data):
            # Added to handle geometric Data and DataBatch types
            x = data.x
            batch = data.batch
            pos_conv = data.pos
            pos_graph = data.pos_graph

            if self.t_scale is not None: # rescale by time scaling factor
                pos_graph[:,3] = pos_graph[:,3] / self.t_scale
            sa0_out = (x, pos_conv, batch, pos_graph)
        else:
            # Legacy solution (padded point clouds)
            batchsize = data.shape[0]
            npoints = data.shape[1]
            x = data.reshape((batchsize * npoints, 3))
            batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
            sa0_out = (x, x, batch)
        
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        y = self.mlp(x)

        return y