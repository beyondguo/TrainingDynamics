# TrainingDynamics

> This project is mainly inspired by [AllenAI's Dataset Cartography](https://github.com/allenai/cartography) project, where the model outputs (logits) of each sample is recorded after every training epoch. Based on these records, training dynamics (prediction confidence, variability, etc.) are computed to plot the Data Cartography to visualize the distribution of all training samples. However, the [original repo](https://github.com/allenai/cartography) hasn't been maintained for a long time. In this repo, we use the latest version of packages to reimplement the Dataset Cartography, as well as some other extensions based on the training dynamics.

Basic requirements:
- transformers==4.18.0
- torch==1.7.0
- datasets==2.3.2
- accelerate==0.9.0
More requirements see `requirements.txt`.

