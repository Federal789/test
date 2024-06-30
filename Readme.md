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
  - [Command](#Command)
  - [Process](#Process)
- [Results](#Results)
  - [Visualization](#Visualization)
- [Authors](#Authors)

<!-- /TOC -->

# Brief
This project is based on WA-SSGAN model to study lithology identification with few labels.
- [Github](https://github.com/Federal789/test/tree/master)


# Environment

## Configuration

Hardware：Ubuntu 18.04.6 LTS + NVIDIA GeForce RTX 2080ti(12GB)；

Language：Python 3.6.0;

Frame：Tensorflow-gpu==1.8.0, Keras==2.1.6;

Compile：gcc==7.5.0, g++==7.5.0：

Driver：DRIVER VERSION:470.141.03 + CUDA-11.2 + CUDNN-8.0.5。


## Version
keras==2.1.6; \
tensorflow-gpu==1.8.0; \
protobuf==3.17.2; \
tfdeterminism==0.1.0; \
numpy==1.19.2; \
protobuf==3.17.2; \
pandas==0.25.3; \
matplotlib==3.3.4

# Dataset
The log datasets were obtained from five different wells (A-E) in the Jiyang Depression, Bohai Bay Basin.
The origin data is in composed of single log point.
![image](.\image\fig4.png)

We combine continuous samples into log segments and divide them randomly, as follows:
### Create the labelled samples
```python
def batch_labeled(self, batch_size, X, Y):
    idx = np.random.randint(0, len(Y), batch_size)
    samples, labels = X[idx], Y[idx]
    return samples, labels, idx
```
### Create the unlabelled samples
```python
def batch_unlabeled(self, batch_size):
    X, Y = self.A_test, self.Y_test
    idx = np.random.randint(0, np.shape(X)[0], batch_size)
    samples, labels = X[idx], Y[idx]
    return samples, labels
```
### Create the test set
```python
def test_set(self):
    return self.A_test, self.Y_test
```


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
        ├── two_compare.png                // The comparisive results of several methods under various label ratios
    ├── image                              // Several images of results and structures
        ├── fig1.png                       // WA algorithm schematic diagram
        ├── fig2.png                       // Structure of WA-SSGAN
        ├── fig3.png                       // Generated log visualization
        ├── fig4.png                       // Visualization of the well logs
    ├── main.py                            // Process the dataset and define the basic architecture of the model
    ├── KF1.py                             // Construct the first submodel and visualize the intermediate results 
    ├── KF2.py                             // Construct the second submodel and visualize the intermediate results
    ├── KF3.py                             // Construct the third submodel and visualize the intermediate results
    ├── train.py                           // Train the overall model and plot the confusion matrices
    ├── readme.md                          // Readme document
```


# Execute
## Command
1.Install the dependency package. \
2.Run 'cd SSGAN_COM'. \
3.Run 'python train.py'. \
4.The documents in directories 'COMP_EX' and 'draw' could be run independently.
5.If you would like to plot the results of the training process of a certain subnetwork, the commands below can be performed directly:\
(1)python KF1.py \
(2)python KF2.py \
(3)python KF3.py


## Process
### The backbone network
![image](.\image\fig2.png)

### Set a subnetwork training strategy
```python
if (iteration >= 1) & (fscore_val > best_sc):
    dis_wb = discriminator_supervised.get_weights()
    gen_wb = generator.get_weights()
    y_te_1 = np.argmax(discriminator_supervised.predict(x_test), axis=1)
    test_score_1 = f1_score(y_test, y_te_1, average='macro')
    best_para = dis_wb * 1
    best_gen_para = gen_wb
    best_sc = fscore_val * 1
    best_test = test_score_1 * 1
    best_epoch = iteration + 1
```

### Weighted the parameters of three sub-networks
```python
for i in range(len_param):
    new_pa = (dis_wb1[i] * (fscore1 ** 1) + dis_wb2[i] * (fscore2 ** 1) + dis_wb3[i] * (fscore3 ** 1))\
            / (1 * (fscore1 ** 1) + 1 * (fscore2 ** 1) + 1 * (fscore3 ** 1))
    new_param.append(new_pa)

net_new = build_discriminator(sam_shape, dropout)
NEW_classifier = build_discriminator_supervised(net_new)
NEW_classifier.set_weights(new_param)
```
## Results

#### 

#### The generated well logs are shown as follows.
![image](.\image\fig3.png) \
More results are visualized in the manuscript.

## Authors
Jichen Wang, Jing Li, Kun Li, Zerui Li, Yu Kang, Ji Chang*, Wenjun Lv*