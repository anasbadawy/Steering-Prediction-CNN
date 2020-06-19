import numpy as np
import os
import pandas as pd
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from sklearn import model_selection,utils
from dataPreprocessing import generate_samples, preprocess


if __name__ == '__main__':

    # Read the data
    df = pd.read_csv('driving_log_balanced.csv')

    # Split data into 80% train and 20% validation sets
    df = utils.shuffle(df,random_state=0)
    df_train, df_valid = model_selection.train_test_split(df, test_size=.2)

    df_train.to_csv('train.csv', index=False)
    df_valid.to_csv('test.csv', index=False)


    # CNN Model Architecture
    model = models.Sequential()
    model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(core.Flatten())
    model.add(core.Dense(500, activation='relu'))
    model.add(core.Dropout(.5))
    model.add(core.Dense(100, activation='relu'))
    model.add(core.Dropout(.25))
    model.add(core.Dense(20, activation='relu'))
    model.add(core.Dense(1))
    model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')# Adam optimizer with learning rate of 1e-04 and loss func mse

    history = model.fit_generator( # start training model for 20 epochs
        generate_samples(df_train, ''),
        samples_per_epoch=df_train.shape[0],
        nb_epoch=20,
        validation_data=generate_samples(df_valid, '', augment=False),
        nb_val_samples=df_valid.shape[0],
    )

    with open(os.path.join('', 'model.json'), 'w') as file:# save trained model 
        file.write(model.to_json())

    backend.clear_session()