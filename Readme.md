# __Contents__

<!-- TOC -->
- [Brief](#Brief)
- [Environment](#Environment)
  - [Configuration](#Configuration)
  - [Version](#Version)
- [Dataset](#Dataset)
- [Exhibition](#Exhibition)
  - [Structure](#Structure)
- [Execute](#Execute)
- [Results](#Results)
  - [Visualization](#Visualization)
- [Authors](#Authors)

<!-- /TOC -->

# Project Brief
This project is based on WA-SSGAN model to study lithology identification with few labels.
- [Github](https://github.com/Federal789/test)


# Environment

## Configuration

Hardware：Ubuntu 18.04.6 LTS + NVIDIA GeForce RTX 2080ti(12GB)；

Language：Python 3.6.0;

Frame：Tensorflow-gpu==1.8.0, Keras==2.1.6;

Compile：gcc==7.5.0, g++==7.5.0：

Driver：DRIVER VERSION:470.141.03 + CUDA-11.2 + CUDNN-8.0.5。


## Library Version
keras==2.1.6; \
tensorflow-gpu==1.8.0; \
protobuf==3.17.2; \
tfdeterminism==0.1.0; \
numpy==1.19.2; \
protobuf==3.17.2; \
pandas==0.25.3; \
matplotlib==3.3.4

# Dataset
The log datasets were obtained from five different wells (A-E) in the Jiyang
Depression, Bohai Bay Basin.
We're sorry that because it involves non-public information, the data sets are confidential.

# Exhibition
## Structure
```bash
├── SSGAN_COM
    ├── COMP_EX                            // Comparasive experiments
        ├── FCN.py                         // Comparasive experiment with FCN model
        ├── SUP.py                         // Comparasive experiment with the discriminator
    ├── draw                               // Codes of visualizing the prediction results
        ├── compare.py                     // Comparison of WA-SSGAN and the discriminator
        ├── draw_3d.py                     // Results of each lithology under various label ratios
        ├── other_model.py                 // Comparasive experiments visualizations
        ├── radar.py                       // Radar maps
        ├── trend.py                       // The change trend of prediction results 
        ├── tSNE.py                        // Visual intersection of five wells
    ├── image                              // Several images of results and structures
        ├── fig1.png                       // WA algorithm schematic diagram
        ├── fig2.png                       // Structure of WA-SSGAN
        ├── fig3.png                       // Generated log visualization
    ├── KF1.py                             // Construct the first submodel and visualize the intermediate results 
    ├── KF2.py                             // Construct the second submodel and visualize the intermediate results
    ├── KF3.py                             // Construct the third submodel and visualize the intermediate results
    ├── train.py                           // Train the overall model and plot the confusion matrices
    ├── readme.md                          // Readme document
```


## Execute
1.Install the dependency package. \
2.Run 'cd SSGAN_COM'. \
3.Run 'python train.py'. \
4.The documents in directories 'COMP_EX' and 'draw' could be run independently.

## Results
The generated well logs are shown as follows.
![image](.\image\fig3.png)
More visualizations are listed in the article.

## Authors
Jichen Wang, Jing Li, Kun Li, Zerui Li, Yu Kang, Wenjun Lv*