# SlideGraph
Couple whole slide image (WSI)-level graph representation with a graph convolutional network for capturing the global context of a WSI. This pipeline generates node-level and WSI-level predictions by using a graph representation to capture the biological geometric structure of the cellular architecture at the entire WSI level.

## Environment
Please refer to requirements.txt

## Training the classification model
Before training, it is required to generate x,y coordinates, feature vectors for local regions in the WSIs. Feature varies. It can be nuclear composition features (e.g.,counts of different types of nuclei in the patch), morphological features, receptor expression features, deep features (or neuralfeature embdeddings from a pre-trained neural network) and so on. 

Each WSI should be fitted with one npz file which contains three parts: x_coordinate, y_coordinate and corresponding patch-level feature vector. Please refer to feature.npz in the example folder.

Then, run features_to_graph.py to group spatially neighbouring regions with high degree of feature similarity and construct a graph based on clusters for each WSI.

After getting graphs of all WSIs, train the classification model and do 5-fold stratified cross validation using

python train.py

Parameters learning_rate, weight_decay can be adjusted in train.py
