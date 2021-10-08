# SlideGraph+: Whole Slide Image Level Graphs to Predict HER2 Status in Breast Cancer
A novel graph neural network (GNN) based model (termed SlideGraph+) to predict HER2 status directly from whole-slide images of routine Haematoxylin and Eosin (H&E) slides. This pipeline generates node-level and WSI-level predictions by using a graph representation to capture the biological geometric structure of the cellular architecture at the entire WSI level. A pre-processing function is used to do adaptive spatial agglomerative clustering to group spatially neighbouring regions with high degree of feature similarity and construct a WSI-level graph based on clusters.

## Data
The repository can be used for constructing WSI-level graphs, training SlideGraph and predicting HER2 status on WSI-level graphs. The training data used in this study was downloaded from TCGA using https://portal.gdc.cancer.gov/projects/TCGA-BRCA.

## Workflow of predicting HER2 status from H&E images
<img width="1201" alt="workflow1" src="https://user-images.githubusercontent.com/58427109/136570069-57686d6c-e34f-4176-a8ec-9c86400f7cc7.png">

## GNN network architecture
![GCN_architecture5](https://user-images.githubusercontent.com/58427109/136584825-8866d382-5e9d-48b9-99e9-20e88f87b804.png)

## Environment
Please refer to requirements.txt

## Repository Structure
Below are the main executable scripts in the repository:

features_to_graph.py: Construct WSI-level graph 

platt.py: Normalise classifier output scores to a probability value 

GNN_pr.py: Graph neural network architecture

train.py: Main training and inference script

## Training the classification model
### Data format
For training, each WSI has to have a WSI-level graph. In order to do that, it is required to generate x,y coordinates, feature vectors for local regions in the WSIs. x,y coordinates can be cental points of patches, centroid of nuclei and so on. Feature varies. It can be nuclear composition features (e.g.,counts of different types of nuclei in the patch), morphological features, receptor expression features, deep features (or neuralfeature embdeddings from a pre-trained neural network) and so on. 

Each WSI should be fitted with one npz file which contains three arrays: x_coordinate, y_coordinate and corresponding region-level feature vector. Please refer to feature.npz in the example folder.

### Graph construction
After npz files are ready, run features_to_graph.py to group spatially neighbouring regions with high degree of feature similarity and construct a graph based on clusters for each WSI.

* Set path to the feature directories (feature_path) 
* Set path where graphs will be saved (output_path) 
* Modify hyperparameters, including similarity parameters (lambda_d, lambda_f), hierachical clustering distance threshold (lamda_h) and node connection distance threshold (distance_thres) 

### Training
After getting graphs of all WSIs, 
* Set path to the graph directories (bdir) in train.py  
* Set path to the clinical data (clin_path) in train.py  
* Modify hyperparameters, including learning_rate, weight_decay in train.py 

Train the classification model and do 5-fold stratified cross validation using

python train.py

In each fold, top 10 best models (on validation dataset) and the model from the last epoch are tested on the testing dataset. Averaged classification performance among 5 folds are presented in the end.

## Heatmap of node-level prediction scores
<img width="634" alt="heatmap_final" src="https://user-images.githubusercontent.com/58427109/136618599-d41b4653-2e2c-4d5a-ad49-a7de2fc001b3.png">
Heatmaps of node-level prediction scores and zoomed-in regions which have different levels of HER2 prediction score. Boundary colour of each zoomed-in region represents its contribution to HER2 positivity (prediction score).

## License

The source code SlideGraph as hosted on GitHub is released under the [GNU General Public License (Version 3)].

The full text of the licence is included in [LICENSE.md](https://github.com/wenqi006/SlideGraph/blob/main/LICENSE.md).

[gnu general public license (version 3)]: https://www.gnu.org/licenses/gpl-3.0.html
