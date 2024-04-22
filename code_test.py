from PIL import Image
import numpy as np
from datetime import datetime
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="kkkl.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
# Load and preprocess the image
img_name = '20240109_203301.jpg'
img = Image.open(img_name)
im_w, im_h = img.size  # Get dimensions before conversion
img = img.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
img = np.array(img).astype(np.float32)  # Convert image to float32 numpy array
 
# Add batch dimension
input_data = np.expand_dims(img, axis=0)
print("Image path:", img_name)
print("Input data shape:", input_data.shape)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)
# Perform inference
startTime = datetime.now()
interpreter.invoke()
delta = datetime.now() - startTime
print("Inference time:", '%.1f' % (delta.total_seconds() * 1000), "ms")

# Get output data
# Get output data
output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)

# Assuming results are structured as scores followed by bounding boxes
# Example structure: [score1, score2, score3, ymin, xmin, ymax, xmax]
scores = results[:, :3]  # Assuming first three are scores
boxes = results[:, 3:]  # Assuming last four are bounding box coordinates

# Apply a sigmoid function if scores are logits
scores = 1 / (1 + np.exp(-scores))

# Sort scores and select top indices
top_indices = np.argsort(scores, axis=0)[:, ::-1]

# Print the top detection results
labels = ['Background', 'Apple', 'Orange']
for label_idx, label in enumerate(labels):
    print(f"Top 5 detections for {label}:")
    for i in range(5):
        index = top_indices[i, label_idx]
        score = scores[index, label_idx]
        ymin, xmin, ymax, xmax = boxes[index]
        # Calculate bounding box positions on the image
        left = int(xmin * im_w)
        right = int(xmax * im_w)
        top = int(ymin * im_h)
        bottom = int(ymax * im_h)
        print(f'Score: {score:.2f}. Location: ({left}, {top}, {right}, {bottom})')
