import numpy as np
import tensorflow as tf
import cv2

# Paths for the image and model
path = "orange_003.jpg"
model_path = "good.tflite"

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to load and preprocess the image
def load_and_preprocess_image(path):
    img = cv2.imread(path)
    #img = cv2.rotate(img, cv2.ROTATE_180)  # Rotate the image to the correct orientation

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Keep original orientation, remove rotation
    img = cv2.resize(img, (320, 320))
    img = (img / 255.0)    *2-1  # Normalize the image to the range [-1, 1]
    img = img.astype(np.float32)

    return np.expand_dims(img, axis=0)

# Load and preprocess the input image
image = load_and_preprocess_image(path)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], image)

# Run the inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Assume scores are the first 3 entries and boxes the next 4 entries of the last dimension
scores = output_data[0, :, :3]
boxes = output_data[0, :, 3:7]

# Normalize scores using softmax
scores = tf.nn.softmax(scores, axis=-1).numpy()
boxes = tf.sigmoid(boxes).numpy()
# Load the image for drawing boxes
image = cv2.imread(path)
image = cv2.resize(image, (320, 320))  # Ensure image is resized but not rotated
print(scores)
# Draw the bounding boxes on the image
for i, score in enumerate(scores):
    
    label = np.argmax(score[1:]) + 1  # Adjusting for class indexing
    confidence = score[label]
    if confidence > 0.99:
        print(boxes[i])
        cx, cy, h, w = boxes[i]
        xmin = int((cx - w / 2) * 320)
        xmax = int((cx + w / 2) * 320)
        ymin = int((cy - h / 2) * 320)
        ymax = int((cy + h / 2) * 320)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow("Image with Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
