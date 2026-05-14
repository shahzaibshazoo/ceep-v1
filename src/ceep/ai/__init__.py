"""
neurowave.ai — AI Integration and Differentiable Physics
========================================================

This module provides AI/ML integration capabilities:

- **Differentiable**: Differentiable Maxwell solver (PyTorch autograd)
- **Inverse**: Inverse problem solvers (gradient-based reconstruction)
- **Networks**: Neural network architectures for EM problems
- **PINNs**: Physics-Informed Neural Networks for Maxwell equations
- **Optimization**: AI-driven antenna/device optimization

Architecture
------------
The AI module is designed for seamless PyTorch integration:
- FDTD operations wrapped as torch.autograd.Function
- Field arrays stored as CUDA tensors
- Gradient propagation through simulation timesteps
- Compatible with PyTorch optimizers and schedulers

Target Applications
-------------------
- Microwave brain hemorrhage imaging (PGA-Net, EMNeRF)
- Inverse scattering / diffraction tomography
- Antenna topology optimization
- Metamaterial design
- Learned boundary conditions

References
----------
.. [1] Hughes et al., "Wave physics as an analog recurrent neural network,"
       Science Advances, 2019.
.. [2] Chen et al., "Physics-informed neural networks for inverse problems
       in nano-optics and metamaterials," Opt. Express, 2020.
"""
