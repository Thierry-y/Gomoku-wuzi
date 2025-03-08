from torchview import draw_graph
import torch
from pol_val_net import ConvBlock, ResidueBlock, PolicyHead, ValueHead, PolicyValueNet  

dummy_input = torch.randn(1, 6, 9, 9)  

# ConvBlock
conv_block = ConvBlock(in_channels=6, out_channel=128, kernel_size=3, padding=1)
graph = draw_graph(conv_block, input_data=dummy_input, expand_nested=True)
graph.visual_graph.render("../img/conv_block", format="png")

# ResidueBlock
dummy_input_128 = torch.randn(1, 128, 9, 9)  
residue_block = ResidueBlock(in_channels=128, out_channels=128)
graph = draw_graph(residue_block, input_data=dummy_input_128, expand_nested=True)
graph.visual_graph.render("../img/residue_block", format="png")

# PolicyHead
policy_head = PolicyHead(in_channels=128, board_len=9)
graph = draw_graph(policy_head, input_data=dummy_input_128, expand_nested=True)
graph.visual_graph.render("../img/policy_head", format="png")

# ValueHead
value_head = ValueHead(in_channels=128, board_len=9)
graph = draw_graph(value_head, input_data=dummy_input_128, expand_nested=True)
graph.visual_graph.render("../img/value_head", format="png")

# PolicyValueNet
# model = PolicyValueNet(board_len=9, n_feature_planes=6, is_use_gpu=False)
# graph = draw_graph(model, input_data=dummy_input, expand_nested=True)
# graph.visual_graph.render("../img/policy_value_net_structure", format="png")
