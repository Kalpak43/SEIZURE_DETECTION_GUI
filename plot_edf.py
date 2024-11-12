import base64
import mne
import io 
import matplotlib
matplotlib.use("Agg")  # Set backend to non-interactive
import matplotlib.pyplot as plt

def plot_edf_with_mne(file_path):
    try:
        print("-------------------------Plotting started")
        
        # Read and plot the EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True)
        fig, ax = plt.subplots()
        raw.plot(scalings='auto', show=False, title='EDF File')

        # Save the plot to a buffer
        img_buf = io.BytesIO()  # Use BytesIO directly
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)

        # Encode the buffer to base64
        img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close(fig)

        return img_base64  # Return base64 encoded image
    except Exception as e:
        print(f"Error in plotting: {e}")
        return None