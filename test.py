import time
import mne
import numpy as np
from tensorflow.keras.models import load_model

def process_segement(segment):
    # Apply z-score normalization
    data = segment.get_data()
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    zscored_data = (data - mean) / std
    
    # zscored_data = data
    
    segment_features = []

    # Loop through each channel
    for channel in range(zscored_data.shape[0]):
        # Extract the signal for this specific channel
        signal = zscored_data[channel, :]
        
        # Calculate features for the current channel
        mean_val = np.mean(signal)
        variance_val = np.var(signal)
        std_dev_val = np.std(signal)
        
        # Append all features for this channel (5 features in total)
        segment_features.extend([mean_val, variance_val, std_dev_val])

    
    # Convert the list of features to a NumPy array
    features = np.array(segment_features)
    
    # Apply z-normalization (standardization)
    mean = np.mean(features, axis=0)  # Mean of each feature
    std = np.std(features, axis=0)    # Standard deviation of each feature

    # Perform z-normalization (subtract the mean and divide by std deviation)
    normalized_features = (features - mean) / std
    
    print(normalized_features)
    
    # Convert to a NumPy array and reshape to match model's input shape
    return normalized_features.reshape(1, 57, 1)
    
    
    
def predict(features): 
    model = load_model('my_model.h5')
    prediction = model.predict(features)
    print(prediction)
    class_weights = {0: 0.5, 1: 0.6418918918918919, 2: 2.5966666666666667, 3: 17.545045045045047}

    # Normalize the weights to sum to 1
    total_weight = sum(class_weights.values())
    normalized_weights = {key: value / total_weight for key, value in class_weights.items()}

    threshold = 0.5  # Use class 0's weight to set the threshold

    # Get the probability of the first class (class 0)
    first_class_probability = prediction[0][0]

    # Get the maximum probability from the remaining classes
    remaining_class_probabilities = prediction[0][1:]  # Exclude the first class
    max_remaining_probability = np.max(remaining_class_probabilities)

    print(first_class_probability - max_remaining_probability)

    # Check the difference between the first class and the maximum of the other classes
    if first_class_probability - max_remaining_probability > threshold:
        predicted_class = 0  # Predict as class 0 (first class)
    else:
        # Otherwise, predict the class with the highest probability from the remaining classes
        predicted_class = np.argmax(remaining_class_probabilities) + 1  # Add 1 to account for excluding class 0
    
    return predicted_class



# raw = mne.io.read_raw_edf("uploads/p11_Record2.edf", preload=True)
# segments\595294ca-b245-475b-b63f-1cc0198323d9\segment_1.fif

segment_file_path = "segments/595294ca-b245-475b-b63f-1cc0198323d9/segment_1.fif"

raw = mne.io.read_raw_fif(segment_file_path, preload=True)
    
# Select the first 19 channels
if len(raw.ch_names) > 19:
    raw.pick_channels(raw.ch_names[:19])

total_duration = raw.times[-1]

segment = raw

# segment = raw.copy().crop(tmin=10225, tmax=min(10230, total_duration), include_tmax=False)


features = process_segement(segment)


predicted_class = predict(features)


print(predicted_class)









# [-1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678]



# [-1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678 -1.41421356  0.70710678  0.70710678
#  -1.41421356  0.70710678  0.70710678]


# [[ 0.48017075  0.46364778  0.41407886 ... -0.13117917 -0.07169647
#    0.00761378]
#  [ 0.00593232  0.10408458  0.17769878 ...  0.4132642   0.37400329
#    0.32492716]
#  [ 0.99132294  0.9320238   0.86531227 ... -0.40220677 -0.36885101
#   -0.31696426]
#  ...
#  [-0.32882581 -0.3250068  -0.30591176 ...  0.030161    0.08362712
#    0.13709324]
#  [-0.31259458 -0.30779228 -0.29338541 ...  0.16283225  0.12441392
#    0.09560017]
#  [ 0.49447021  0.47285163  0.45123305 ...  0.17019154  0.17019154
#    0.22063488]]


# [[-0.01312077  0.07283147  0.13649979 ...  0.13968321  0.1587837
#    0.16515054]
#  [ 0.07439713  0.1011557   0.15467282 ... -0.09507377 -0.03858347
#   -0.03263712]
#  [-0.41193271 -0.34888288 -0.31978295 ... -0.1451834  -0.08213357
#   -0.03848368]
#  ...
#  [ 0.19327681  0.22011932  0.22906682 ...  0.59591445  0.58696695
#    0.5824932 ]
#  [ 0.07748783  0.04787033  0.01825284 ...  0.79253878  0.745997
#    0.72061058]
#  [ 0.24116763  0.30049429  0.33015762 ...  0.66535326  0.67721859
#    0.70688192]]


# [ 4.54747351e-17  1.00000000e+00  1.00000000e+00  1.13686838e-17
#   1.00000000e+00  1.00000000e+00  5.11590770e-17  1.00000000e+00
#   1.00000000e+00  1.13686838e-17  1.00000000e+00  1.00000000e+00
#   0.00000000e+00  1.00000000e+00  1.00000000e+00 -5.68434189e-18
#   1.00000000e+00  1.00000000e+00 -3.41060513e-17  1.00000000e+00
#   1.00000000e+00 -1.13686838e-17  1.00000000e+00  1.00000000e+00
#  -1.13686838e-17  1.00000000e+00  1.00000000e+00  2.84217094e-18
#   1.00000000e+00  1.00000000e+00 -2.27373675e-17  1.00000000e+00
#   1.00000000e+00 -1.13686838e-17  1.00000000e+00  1.00000000e+00
#   3.41060513e-17  1.00000000e+00  1.00000000e+00  0.00000000e+00
#   1.00000000e+00  1.00000000e+00 -3.41060513e-17  1.00000000e+00
#   1.00000000e+00  2.84217094e-17  1.00000000e+00  1.00000000e+00
#  -2.84217094e-17  1.00000000e+00  1.00000000e+00  4.54747351e-17
#   1.00000000e+00  1.00000000e+00 -1.13686838e-17  1.00000000e+00
#   1.00000000e+00]


# [-1.13686838e-17  1.00000000e+00  1.00000000e+00  2.84217094e-18
#   1.00000000e+00  1.00000000e+00 -5.68434189e-17  1.00000000e+00
#   1.00000000e+00 -1.70530257e-17  1.00000000e+00  1.00000000e+00
#   5.68434189e-18  1.00000000e+00  1.00000000e+00 -5.68434189e-18
#   1.00000000e+00  1.00000000e+00 -2.84217094e-18  1.00000000e+00
#   1.00000000e+00  2.84217094e-17  1.00000000e+00  1.00000000e+00
#  -2.27373675e-17  1.00000000e+00  1.00000000e+00  1.13686838e-17
#   1.00000000e+00  1.00000000e+00 -1.13686838e-17  1.00000000e+00
#   1.00000000e+00  0.00000000e+00  1.00000000e+00  1.00000000e+00
#   2.84217094e-17  1.00000000e+00  1.00000000e+00 -5.68434189e-18
#   1.00000000e+00  1.00000000e+00  0.00000000e+00  1.00000000e+00
#   1.00000000e+00  8.52651283e-18  1.00000000e+00  1.00000000e+00
#  -2.84217094e-17  1.00000000e+00  1.00000000e+00 -8.52651283e-18
#   1.00000000e+00  1.00000000e+00 -1.42108547e-17  1.00000000e+00
#   1.00000000e+00]