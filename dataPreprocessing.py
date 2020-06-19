import numpy as np
import skimage.transform as sktransform
import random
from PIL import Image


def preprocess(frame, top_offset=.375, bottom_offset=.125):
    
    #crops `top_offset` and `bottom_offset` portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    image = sktransform.resize(frame[int(top_offset * frame.shape[0]):-int(bottom_offset * frame.shape[0]), :], (32, 128, 3))
    return image


def generate_samples(data, root_path, augment=True):

    # genrate batch of data
    while True:
        # Generate random batch of indices
        indices = np.random.permutation(data.count()[0])
        batch_size = 128
        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]
            # Output arrays
            x = np.empty([0, 32, 128, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            # Read in and preprocess a batch of images
            for i in batch_indices:
                
                # Read frame image and work out steering angle
                image = Image.open(data["images"].values[i].strip())
                image = np.asarray(image)
                image = image.copy()
                angle = data.steering.values[i]
                if augment:
                    # Add random shadow as a vertical slice of image
                    h, w = image.shape[0], image.shape[1]
                    [x1, x2] = np.random.choice(w, 2, replace=False)
                    k = h / (x2 - x1)
                    b = - k * x1
                    for i in range(h):
                        c = int((i - b) / k)
                        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
                # Randomly shift up and down while preprocessing
                v_delta = .05 if augment else 0
                image = preprocess(
                    image,
                    top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
                    bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta)
                )
                # Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])
            # Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)
