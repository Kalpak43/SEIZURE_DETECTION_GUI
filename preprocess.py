import json
from pathlib import Path
import time
import mne
import numpy as np
from keras.models import load_model
import uuid



# Function to segment EEG data into 5-second intervals
def segment_eeg(file_path, segment_duration=5):
    
    # Generate a unique identifier for the segment
    segments_id = uuid.uuid4()
    segments_dir = Path(f'segments/{segments_id}')
    segments_dir.mkdir(parents=True, exist_ok=True)

    
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    # Select the first 19 channels
    if len(raw.ch_names) > 19:
        raw.pick_channels(raw.ch_names[:19])
    
    total_duration = raw.times[-1]
        
    # Initialize variables for segmentation
    segment_start = 0
    segment_end = segment_duration

    # Total number of segments
    total_segments = int(total_duration // segment_duration) + 1

    # Segment the data
    current_segment = 0
    
    consecutive_seizure_count = 0
    consecutive_non_seizure_count = 0
    seizure_events = []
    non_seizure_events = []
    seizure_started = False
    non_seizure_started = False
    
    while segment_start < total_duration:
        print(f"Segment Start: {segment_start}, Segment End: {segment_end}")
        
        start_sample = int(segment_start * raw.info['sfreq'])
        end_sample = int(segment_end * raw.info['sfreq'])
        
        print(f"Segment {current_segment + 1}/{total_segments}: {segment_start:.2f}s - {min(segment_end, total_duration):.2f}s")    
        
        if end_sample > len(raw.times):
            end_sample = len(raw.times)

        # Extract the segment without altering the original raw object
        segment = raw.copy().crop(tmin=segment_start, tmax=min(segment_end, total_duration), include_tmax=False)
        
        # Save the segment to disk
        segment_file_path = segments_dir / f"segment_{current_segment + 1}.fif"
        segment.save(segment_file_path, overwrite=True)
        
        features = process_segement(segment)
        predicted_class = predict(features)
        
        # Calculate progress percentage
        progress_percentage = (current_segment + 1) / total_segments * 100

        # Seizure and non-seizure tracking logic
        if predicted_class >= 1:
            consecutive_seizure_count += 1
            consecutive_non_seizure_count = 0  # Reset non-seizure count
            
            if consecutive_seizure_count >= 2:
                if not seizure_started:
                    seizure_started = True
                    seizure_events.append({"start": segment_start})
                non_seizure_started = False

        else:
            consecutive_non_seizure_count += 1
            consecutive_seizure_count = 0  # Reset seizure count

            if consecutive_non_seizure_count >= 2:
                if not non_seizure_started:
                    non_seizure_started = True
                    non_seizure_events.append({"start": segment_start})
                seizure_started = False

        # Update end time for ongoing events
        if seizure_started:
            seizure_events[-1]["end"] = segment_end
        if non_seizure_started:
            non_seizure_events[-1]["end"] = segment_end

        # Yield predicted class, progress, and event data
        yield f"data: {{\"predicted_class\": {predicted_class}, \"progress\": {progress_percentage:.2f}, \"seizure_events\": {json.dumps(seizure_events)}, \"non_seizure_events\": {json.dumps(non_seizure_events)}}}\n\n"
        
        # Increment the segment counter
        current_segment += 1
        
        # Move to the next segment
        segment_start += segment_duration
        segment_end += segment_duration

        
        


def process_segement(segment):
    # Apply z-score normalization
    data = segment.get_data()
    # mean = np.mean(data, axis=1, keepdims=True)
    # std = np.std(data, axis=1, keepdims=True)
    # zscored_data = (data - mean) / std

    
    segment_features = []

    # Loop through each channel
    for channel in range(data.shape[0]):
        # Extract the signal for this specific channel
        signal = data[channel, :]
        
        # Calculate features for the current channel
        mean_val = np.mean(signal)
        variance_val = np.var(signal)
        std_dev_val = np.std(signal)
        
        # Append all features for this channel (5 features in total)
        segment_features.extend([mean_val, variance_val, std_dev_val])

    
    # Convert the list of features to a NumPy array
    features = np.array(segment_features)
    
    print("-----------------------------------------------------------------------")
    print(features)
    
    # Apply z-normalization (standardization)
    mean = np.mean(features, axis=0)  # Mean of each feature
    std = np.std(features, axis=0)    # Standard deviation of each feature

    # Perform z-normalization (subtract the mean and divide by std deviation)
    normalized_features = (features - mean) / std
    
    # Convert to a NumPy array and reshape to match model's input shape
    
    print(normalized_features)
    
    return normalized_features.reshape(1, 57, 1)
    
    
    
def predict(features): 
    model = load_model('my_model.h5')
    prediction = model.predict(features)
    print(prediction)
    class_weights = {0: 0.5, 1: 0.6418918918918919, 2: 2.5966666666666667, 3: 17.545045045045047}

    # Normalize the weights to sum to 1
    total_weight = sum(class_weights.values())
    normalized_weights = {key: value / total_weight for key, value in class_weights.items()}

    threshold = normalized_weights[0]  # Use class 0's weight to set the threshold

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
    
    # predicted_class = np.argmax(prediction)
    
    return predicted_class