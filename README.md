Neural Atari Enduro
This project extends the Neural Atari framework to the Atari game Enduro, adapting a VQ-VAE world model and action-to-latent mapping from the original Breakout implementation. A DQN agent was also trained to generate gameplay data. The project includes a playable neural simulation, though it currently outputs a black background due to low action-to-latent accuracy (43.5%).
Project Overview

Objective: Create a neural simulation of Enduro using a VQ-VAE to encode game frames into a 5x7 latent grid (256 codebook) and an ActionToLatentMLP to map 9 actions to latent codes.
DQN Training: Trained a DQN agent for 1000 episodes, achieving high rewards (up to 388) and quality gameplay videos.
Results: Action-to-latent model achieved 43.5% accuracy, leading to a black background in neural gameplay. DQN performance was robust.
Analysis: Visualizations of latent changes and action distributions reveal action imbalances and latent dynamics.

Setup Instructions

Clone the repository:git clone https://github.com/nevernever69/atari-pixels
cd atari-pixels
git checkout dev-enduro


Install dependencies:pip install torch==2.3.0 gymnasium[atari] opencv-python numpy pygame imageio


Generate initial frame:python generate_initial_frame.py


Run the neural simulation:python play_neural_enduro.py --temperature 0.01

Repository Structure

play_neural_enduro.py: Playable neural Enduro simulation (currently displays black background).
latent_action_model.py: VQ-VAE and ActionToLatentMLP model definitions.
atari_env.py: Enduro environment wrapper for Gymnasium.
generate_initial_frame.py: Generates initial game frame (data/0.png).
notebooks/analysis.ipynb: Jupyter notebook with visualizations and analysis.
checkpoints/dqn_latest.pth: DQN model checkpoint.
checkpoints/latent_action/best.pt: VQ-VAE checkpoint.
checkpoints/action_to_latent_best.pt: Action-to-latent model checkpoint.
data/actions/action_latent_pairs.json: 100,000 action-latent pairs.
data/neural_enduro_game.gif: Sample gameplay GIF (DQN-based).

Key Results

DQN Performance: Achieved rewards up to 388 (e.g., episode 974), with coherent gameplay videos.
Action-to-Latent Model: 43.5% validation accuracy, causing black output in neural simulation, likely due to insufficient data or VQ-VAE decoder issues.
VQ-VAE: Adapted for grayscale 1x210x160 frames, with PSNR 20–23 dB.
Analysis: Latent visualizations show action-specific patterns (e.g., RIGHT/LEFT alter specific latents). Action distribution highlights NOOP dominance.

Known Issues

Black Background: Neural simulation outputs a black screen, likely due to low action-to-latent accuracy (43.5%) producing invalid latents.

Future Improvements

Collect 200,000+ action-latent pairs to improve action-to-latent accuracy.
Retrain VQ-VAE with 1000+ episodes for PSNR >25 dB.
Implement ActionStateToLatentMLP to incorporate frame context.
Debug black screen by analyzing VQ-VAE decoder outputs.
Test models on other racing games (e.g., Road Fighter).


Acknowledgments
This project builds on Paras's Neural Atari framework for Breakout. Thanks to the open-source community for tools like Gymnasium, PyTorch, and PyGame.

