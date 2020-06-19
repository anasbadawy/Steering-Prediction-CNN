import argparse
import pandas as pd
from subprocess import call
import numpy as np
from keras.models import model_from_json
from keras import losses
import scipy.misc
from dataPreprocessing import preprocess

model = None


def eval_model(data):
    # The current steering angle of the car
    pred = []
    for i in range(len(data)):
        # The current image from the center camera of the car
        full_image = scipy.misc.imread(data[i], mode="RGB")
        image = preprocess(np.asarray(full_image))
        transformed_image_array = image[None, :, :, :]
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))
        pred.append(round(steering_angle,6))
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    call('clear')
    df = pd.read_csv('test.csv')# read test set 
    y_pred = eval_model(df['images'])
    y_true = []
    for i in range(len(df['steering'])):
        # The current image from the center camera of the car
        y_true.append(round(df['steering'][i], 6))
    

    mae = losses.mean_absolute_error(y_true, y_pred)    
    print(mae)# 0.011395579 epoch 16 wih bins
    # epoch 20 without bins 0.013438199
