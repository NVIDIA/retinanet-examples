import numpy as np
from odtk.box import generate_anchors, generate_anchors_rotated

# Generates anchors for export.cpp

# ratios = [1.0, 2.0, 0.5]
# scales = [4 * 2 ** (i / 3) for i in range(3)]
ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
scales = [2 * 2**(2 * i/3) for i in range(3)]
angles = [-np.pi / 6, 0, np.pi / 6]
strides = [2**i for i in range(3,8)]

axis = str(np.round([generate_anchors(stride, ratios, scales, 
            angles).view(-1).tolist() for stride in strides], decimals=2).tolist()
        ).replace('[', '{').replace(']', '}').replace('}, ', '},\n')

rot = str(np.round([generate_anchors_rotated(stride, ratios, scales, 
            angles)[0].view(-1).tolist() for stride in strides], decimals=2).tolist()
        ).replace('[', '{').replace(']', '}').replace('}, ', '},\n')

print("Axis-aligned:\n"+axis+'\n')
print("Rotated:\n"+rot)
