from .pointnet_tiny import PointNet_Tiny
from .tinytransfomer import TinyTransformer
from .cnn_tiny import CNN_Tiny

model_map = {
    'pointnet_tiny':PointNet_Tiny,
    'transformer_tiny':TinyTransformer,
    'cnn_tiny': CNN_Tiny,
}
