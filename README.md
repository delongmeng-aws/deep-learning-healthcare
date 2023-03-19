# Deep Learning for Healthcare

This is a collection of my deep learning projects for healthcare applications.


[Convolutional Neural Network (CNN) model for pneumonia prediction using chest X-Ray images](https://github.com/delongmeng/deep-learning-healthcare/tree/main/Chest_XRay_CNN)
- build a CNN architecture with convolutional layers and maxpooling layers
- fine tune a pre-trained ResNet18 model for pneumonia classification
- data pre-processing: data leakage, handling imbalance
- fine tune a pre-trained DenseNet121 model for predition of 14 conditions
- evaluation metrics such as ROC curve, sensitivity, specificity, f1 score, PPV, and NPV
- GradCAM visualization for model interpretation
- packages: Python torch, torchvision, keras, sklearn
- keywords: CNN, X-Ray image, radiology diagnosis


[Brain Tumor Auto-Segmentation for Magnetic Resonance Imaging (MRI)](https://github.com/delongmeng/deep-learning-healthcare/blob/main/MRI_Segmentation_3D_UNet/MRI_Auto_Segmentation.ipynb)
- build a neural network based on 3D U-Net to automatically segment tumor regions from MRI scans
- use Dice Similarity Coefficient (DSC) for loss function
- packages: Python keras, tensorflow, nibabel
- keywords: CNN, MRI, segmentation, computer vision


[The Convolutional Attention for Multi-Label classification (CAML) model for medical codes prediction](https://github.com/delongmeng/deep-learning-healthcare/blob/main/CAML_medical_codes_pred/CAML_implementation.ipynb)
- implement the CAML model for disease prediction (20 classes) from clinical notes (chest X-ray reports)
- use a per-label attention mechanism to learn specific representation for each label
- packages: Python torch
- keywords: CNN, attention, clinical notes annotation


[Recurrent Neural Network (RNN) model for heart failure prediction using clinical diagnosis codes](https://github.com/delongmeng/deep-learning-healthcare/blob/main/HeartFailure_diag_codes_RNN/HF_diag_codes_RNN.ipynb)
- build a bi-directional Recurrent Neural Network (RNN) model to predict heart failure using clinical diagnosis codes
- define custom dataset class, collate function, data loader, and the RNN architecture
- packages: Python torch
- keywords: RNN, heart failure, diagnosis codes


[RETAIN, a Recurrent Neural Network (RNN) model with attention mechanism for heart failure prediction](https://github.com/delongmeng/deep-learning-healthcare/blob/main/HeartFailure_RETAIN_RNN_attention/RETAIN_HF_RNN_attention.ipynb)
- build a RNN model with reverse time attention mechanism for heart failure prediction from clinical visits and diagnosis codes
- use 2 attentiona mechnisms for codes within each visit and visits for each patient, respectively
- packages: Python torch, sklearn
- keywords: RNN, attention, heart failure, diagnosis codes


[MINA, a multilevel knowledge-guided attention networks model for heart disease prediction from ECG data](https://github.com/delongmeng/deep-learning-healthcare/blob/main/MINA_ECG_knowledge_guided_attention/MINA_ECG_knowledge_guided_attention.ipynb)
- implement MINA: Multilevel Knowledge-Guided Attention for Modeling Electrocardiography Signals
- this is an advanced CNN+RNN model with prior knowledge-guided attention mechanism to classify ECG recordings to predict heart disease
- define the dataset class, data loader, attention class and the full model
- packages: Python torch
- keywords: CNN, RNN, knowledge-guided attention, ECG, heart disease


[Seq2seq autoencoder model for patient EHR data embedding and unsupervised clustering](https://github.com/delongmeng/deep-learning-healthcare/blob/main/HeartFailure_Seq2Seq_autoencoder/HF_Seq2Seq_autoencoder.ipynb)
- use the sequence-to-sequence (Seq2seq) autoencoder model to generate patient EHR embedding from clinical visits and diagnosis codes
- build a encoder model containing a CNN layer and an attention layer, and a GRU-based decoder model
- build the Seq2seq architecture connecting the encoder and decoder models
- use the embedding from the trained Seq2seq model to conduct unsupervised patient clustering (K-means and t-SNE visualization)
- packages: Python torch, sklearn
- keywords: Seq2seq, autoencoder, embedding, heart failure, diagnosis codes, clustering


[A Graph Convolutional Network (GCN) to classify enzymes](https://github.com/delongmeng/deep-learning-healthcare/blob/main/GCN_graph_chemicals/GCN_graph_chemicals.ipynb)
- implement a graph neural network (GCN) model to predict the mutagenic effect of a certain chemical compound on a specific bacterium
- each graph is a representation of a chemical structure with vertices standing for atoms and edges representing bonds between atoms
- packages: Python torch, torch_geometric
- keywords: GCN


[GAMENet, a graph augmented memory network for medication combination recommendation](https://github.com/delongmeng/deep-learning-healthcare/blob/main/GAMENet_graph_mem_medication_recomm/GAMENet_graph_mem_medication_recomm.ipynb)
- implement GAMENet, a memory network consisting of an input feature map, generalization, an output feature map, and a response component
- given diagnosis codes and procedure codes for the current visit, patient history and EHR graph, predict medication combination of current visit
- create patient representation using 2 RNNs from patient diagnosis and procedure data
- generate and update graph memory bank and dynamic memory table, and make predictions from the output of the memory network
- packages: Python torch, sklearn
- keywords: memory network, GCN, medication combination recommendation


[Word embeddings for medical text](https://github.com/delongmeng/deep-learning-healthcare/blob/main/Word2Vec_Embedding/Word2Vec_Embedding.ipynb)
- train a word2vec model for the NFCorpus dataset
- evaluate the model by checking similar words of a given word
- t-SNE and UMAP visualization of the medical representation
- packages: Python nltk, gensim, sklearn, umap
- keywords: word embedding, medical text


[Heart failure prediction using basic machine learning models](https://github.com/delongmeng/deep-learning-healthcare/blob/main/HeartFailure_Basic_ML/HeartFailure_Basic.ipynb)
- construct feature for patient clinical data of visits and events, and save data into the SVMLight format of sparse representation
- compare different ML models including logistic regression, SVM, and decision tree
- packages: Python numpy, sklearn
- keywords: sparse representation, logistic regression, SVM, decision tree, heart failure


[Heart failure prediction using neural networks](https://github.com/delongmeng/deep-learning-healthcare/blob/main/HeartFailure_NeuralNetwork/HF_NN.ipynb)
- implement a basic neural network model using PyTorch framework for heart failure prediction
- packages: Python torch, numpy, sklearn
- keywords: neural network, heart failure


----

# Machine Learning / Deep Learning Projects outside this repo

## Other healthcare Machine Learning / Deep Learning projects

[Reproduction study for RefDNN: a neural network for cancer drug resistance prediction](https://github.com/delongmeng/DL4H-RefDNN)


[NLP-based disease label extraction using NegBio and clinical question answering using BERT](https://github.com/delongmeng/NLP_label_extraction)


[Cell dimension reduction and clustering: Single-Cell RNA and Protein Profiling](http://htmlpreview.github.io/?https://github.com/delongmeng/SingleCellAnalysis/blob/master/single_cell_analysis.html)


## Other Machine Learning / Deep Learning Projects

see [this repo here](https://github.com/delongmeng/projects)

