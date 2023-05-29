import copy
import torch
from wisp.framework import WispState
from wisp.renderer.core.api import add_to_scene_graph
from wisp.renderer.app.wisp_app import WispApp


# Example for how you bake:
# 1. Load your pretrained hashgrid, see NeRF example on https://kaolin-wisp.readthedocs.io/en/latest/pages/app_nerf.html
hashgrid_saved_model_path = '<CHANGE ME>'
pipeline = torch.load(hashgrid_saved_model_path)   # Load a full pretrained pipeline: model + weights
# 2. Let's duplicate the whole NeRF pipeline, to compare before / after bake:
octree_pipeline = copy.deepcopy(pipeline)
# 3. Call bake to get an Octree:
octree_grid = pipeline.nef.grid.bake()
# 4. The same components of the pipeline remain (NeRF paramters, tracer, etc), only the grid needs to change:
octree_pipeline.nef.grid = octree_grid

# ---- ^^^ It's ready to use ^^^ ----

# ---- vvv Below is some code that shows a side by side comparison of the hash / baked octree ---
# Note: Even without a display, you can view both models in Jupyter:
# https://kaolin-wisp.readthedocs.io/en/latest/pages/example_jupyter.html


# Joint trainer / app state - scene_state contains various global definitions
# If you're using the interactive renderer, this is how you compare both models:
scene_state: WispState = WispState()
add_to_scene_graph(state=scene_state, name='original hash', obj=pipeline, batch_size=2**14)
add_to_scene_graph(state=scene_state, name='baked octree', obj=octree_pipeline, batch_size=2**14)

# !!! If you're using Jupyter, you don't create the app like the lines below. !!!
# Instead have a look at the Jupyter example in the notebook:
scene_state.renderer.device = torch.device('cuda:0')  # Use same device for trainer, models and app renderer
app = WispApp(wisp_state=scene_state, window_name='bake example')
app.run()  # Run in interactive mode


# Finally, this is how I usually tweak my main when I use the full interactive app.
# If you're not using it, you can ditch these lines
# What's interesting here is the new config system which makes the main way shorter:
# https://kaolin-wisp.readthedocs.io/en/latest/pages/config_system.html

# import os
# import logging
# from typing import Optional
# from wisp.app_utils import default_log_setup
# from wisp.config import parse_config, configure, autoconfig, instantiate, print_config
# from wisp.accelstructs import OctreeAS, AxisAlignedBBoxAS
# from wisp.models.grids import OctreeGrid, CodebookOctreeGrid, TriplanarGrid, HashGrid
# from wisp.models.nefs import NeuralRadianceField
# from wisp.models.pipeline import Pipeline
# from wisp.tracers import PackedRFTracer
# from wisp.datasets import NeRFSyntheticDataset, RTMVDataset, SampleRays
# from wisp.trainers import ConfigMultiviewTrainer
# from wisp.trainers.tracker import Tracker, ConfigTracker
# from wisp.renderer.web.jupyter_utils import WISP_ROOT_DIR

# @configure
# class NeRFAppConfig:
#     """ A script for training simple NeRF variants with grid backbones."""
#
#     blas: autoconfig(OctreeAS.make_dense, OctreeAS.from_pointcloud, AxisAlignedBBoxAS)
#     """ Bottom Level Acceleration structure used by the neural field grid to track occupancy, accelerate raymarch. """
#     grid: autoconfig(OctreeGrid, HashGrid.from_geometric, TriplanarGrid, CodebookOctreeGrid)
#     """ Feature grid used by the neural field. Grids are located in `wisp.models.grids` """
#     nef: autoconfig(NeuralRadianceField)
#     """ Neural field configuration, including the feature grid, decoders and optional embedders.
#     NeuralRadianceField maps 3D coordinates (+ 2D view direction) -> RGB + density.
#     Uses spatial feature grids internally for faster feature interpolation and raymarching.
#     """
#     tracer: autoconfig(PackedRFTracer)
#     """ Tracers are responsible for taking input rays, marching them through the neural field to render
#     an output RenderBuffer.
#     """
#     dataset: autoconfig(NeRFSyntheticDataset, RTMVDataset)
#     """ Multiview dataset used by the trainer. """
#     dataset_transform: autoconfig(SampleRays)
#     """ Composition of dataset transforms used online by the dataset to process batches. """
#     trainer: ConfigMultiviewTrainer
#     """ Configuration for trainer used to optimize the neural field. """
#     tracker: ConfigTracker
#     """ Experiments tracker for reporting to tensorboard & wandb, creating visualizations and aggregating metrics. """
#     log_level: int = logging.INFO
#     """ Sets the global output log level: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL """
#     pretrained: Optional[str] = None
#     """ If specified, a pretrained model will be loaded from this path. None will create a new model. """
#     device: str = 'cuda'
#     """ Device used to run the optimization """
#     interactive: bool = os.environ.get('WISP_HEADLESS') != '1'
#     """ Set to --interactive=True for interactive mode which uses the GUI.
#     The default value is set according to the env variable WISP_HEADLESS, if available.
#     Otherwise, interactive mode is on by default. """
#
#
# cfg = parse_config(NeRFAppConfig, yaml_arg='--config')  # Obtain args by priority: cli args > config yaml > config defaults
# cfg.pretrained = os.path.join(WISP_ROOT_DIR, '_results/logs/runs/nerf-hash/20230417-151109/model.pth')
#
# device = torch.device(cfg.device)
# default_log_setup(cfg.log_level)
# if cfg.interactive:
#     cfg.tracer.bg_color = 'black'
#     cfg.trainer.render_every = -1
#     cfg.trainer.save_every = -1
#     cfg.trainer.valid_every = -1
# print_config(cfg)
#
# # Loads a multiview dataset comprising of pairs of images and calibrated cameras:
# # NeRFSyntheticDataset - refers to the standard NeRF format popularized by Mildenhall et al. 2020,
# #   including additions to the metadata format added by Muller et al. 2022.
# # 'rtmv' - refers to the dataset published by Tremblay et. al 2022,
# # RTMVDataset - A Ray-Traced Multi-View Synthetic Dataset for Novel View Synthesis",
# #   this dataset includes depth information which allows for performance improving optimizations in some cases.
# # dataset_transform = instantiate(cfg.dataset_transform)  # SampleRays creates batches of rays from the dataset
# # train_dataset = instantiate(cfg.dataset, transform=dataset_transform)  # A Multiview dataset
# # validation_dataset = None
# # if cfg.trainer.valid_every > -1 or cfg.trainer.mode == 'validate':
# #     validation_dataset = train_dataset.create_split(split='val', transform=None)
#
# if cfg.pretrained and cfg.trainer.model_format == "full":
#     pipeline = torch.load(cfg.pretrained)   # Load a full pretrained pipeline: model + weights
#
# octree_pipeline = copy.deepcopy(pipeline)
# octree_grid = pipeline.nef.grid.bake()
# octree_pipeline.nef.grid = octree_grid
#
# # Joint trainer / app state - scene_state contains various global definitions
# exp_name: str = cfg.trainer.exp_name
# scene_state: WispState = WispState()
#
# add_to_scene_graph(state=scene_state, name='original hash', obj=pipeline, batch_size=2**14)
# add_to_scene_graph(state=scene_state, name='baked octree', obj=octree_pipeline, batch_size=2**14)
#
# test_coords = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.1, -0.1], [0.0, 0.2, -0.1]], device='cuda')
# octree_feats = octree_grid.interpolate(test_coords, lod_idx=0)
# hash_feats = pipeline.nef.grid.interpolate(test_coords, lod_idx=0)
#
# device = torch.device(cfg.device)
# scene_state.renderer.device = device  # Use same device for trainer and app renderer
# app = WispApp(wisp_state=scene_state, window_name='bake example')
#
# app.run()  # Run in interactive mode
