# SlideGraph+: Whole Slide Image Level Graphs to Predict HER2 Status in Breast Cancer
A novel graph neural network (GNN) based model (termed SlideGraph+) to predict HER2 status directly from whole-slide images of routine Haematoxylin and Eosin (H&E) slides. This pipeline generates node-level and WSI-level predictions by using a graph representation to capture the biological geometric structure of the cellular architecture at the entire WSI level.

## Environment
Please refer to requirements.txt

## Data
The data used in this study was downloaded from TCGA using https://portal.gdc.cancer.gov/projects/TCGA-BRCA.

## Training the classification model
Before training, it is required to generate x,y coordinates, feature vectors for local regions in the WSIs. Feature varies. It can be nuclear composition features (e.g.,counts of different types of nuclei in the patch), morphological features, receptor expression features, deep features (or neuralfeature embdeddings from a pre-trained neural network) and so on. 

Each WSI should be fitted with one npz file which contains three parts: x_coordinate, y_coordinate and corresponding region-level feature vector. Please refer to feature.npz in the example folder.

Then, run features_to_graph.py to group spatially neighbouring regions with high degree of feature similarity and construct a graph based on clusters for each WSI.

After getting graphs of all WSIs, train the classification model and do 5-fold stratified cross validation using

python train.py

Parameters learning_rate, weight_decay can be adjusted in train.py

## License

The source code SlideGraph as hosted on GitHub is released under the [GNU General Public License (Version 3)].

The full text of the licence is included in [LICENSE.md](https://github.com/wenqi006/SlideGraph/blob/main/LICENSE.md).

[gnu general public license (version 3)]: https://www.gnu.org/licenses/gpl-3.0.html
