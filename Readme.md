# Readme

## I. Files:

### (i) Dataset file:

#### data_center2.txt: 

The address of the images and the labels are stored in this file.

### (ii) Model_1: (Network: ResNet, Output: Coordinates)

    Dataloader2.py
    model_res.py
    train_res.py
    detect_res.py

### (iii) Model_2: (Network; AlexNet, Output: Coordinates)

    Dataloader2.py
    model_alex.py
    train_alex.py
    detect_alex.py

### (iv) Model_3: (Network: ResNet, Output: Coordinates, K-Fold Validation)

    Dataloader2.py
    model_res.py
    train_res_K_Fold.py
    detect_res.py

### (v) Model_4: (Network: UNet, Output: Heatmap)

    Dataloader_heatmap.py
    model_Unet
    train_heatmap.py
    detect_heatmap.py

### (vi) Model_5: (Network: PFLD, Output: Coordinates, K-Fold Validation)
    
    Dataloader_PFLD.py
    PFlD.py
    train_PFLD_K_Fold.py
    detect_PFLD.py

### (vi) Utilies:
    xml2txt.py
    coordinates2heatmap.py


## II. List of Packages:
```
1. Pytorch
2. Numpy
3. PIL
4. OpenCV
5. TorchVision
6. Math
7. os
8. scipy
9. imgaug
10. heapq
11. Adabelief-pytorch
```

## III. Application:

The organization of the codes and dataset.
```
    first_try
    |-params
    |-part-of-AFLW
    |    --already_labeled
    |        ---***.jpg
    |        ---***.jpg
    |        ---...
    |-***.py
    |-data_center2.txt
    |-test_image
    |    --xxx.jpg
```
There are all 4 models in the project. Each have its own training file, model file and detection file.
1. Training the model: Run the train_xxx.py
2. Use images to test the trained model: Put some images (included in the dataset or not) in the 'test_image' folder and Run detect_xxx.py
3. The image will be shown on the screen, both the ground truth and the prediction.


