# Overview
This repo is personal exploration on how to use Vertex AI Explainable AI features in different settings, with main focus on more custom training/serving situations.

This repository will slowly update with different ML frameoworks/ML problem/XAI metrics. Each will take one separate working folder.

# Examples

### 1. Tensorflow + Classification + Sampled Shapley
The project is inside **example_1** folder:
-   `app/`
    -   `entities.py`: example payload for hosting server
    -   `model.py`: example model architecture
    -   `main.py`: server entrypoint
-   `deploy.py`: example code for Vertex AI platform model creation and online endpoint deployment
-   `Dockerfile`: custom container build
