Step 1: run imageValidDense.py to train DenseNet201 model and generate DenseNet201.hdf5 file.
Step 2: Change "DenseNet201" in imageValidDense.py to "DenseNet169" and generate DenseNet169.hdf5 file.
Step 3: run imageProcDense.py to generate submission file. The submission results are ensemble of DenseNet201 and DenseNet169 models.
