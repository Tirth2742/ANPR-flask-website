import tensorflow as tf
import numpy as np
import cv2
import easyocr
from model_utils import load_model

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the model
detection_model = load_model()

def process_frame(frame):
    """Process the frame for object detection."""
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    image_np_with_detections = image_np.copy()
    detected_plate_text = None

    for i in range(num_detections):
        if detections['detection_scores'][i] > 0.8:
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
            (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1], 
                                          ymin * image_np.shape[0], ymax * image_np.shape[0])
            cv2.rectangle(image_np_with_detections, 
                          (int(left), int(top)), (int(right), int(bottom)), 
                          (0, 255, 0), 2)
            
            cutout = image_np[int(top):int(bottom), int(left):int(right)]
            result = reader.readtext(cutout)
            detected_text = ' '.join([text[1] for text in result])
            
            cv2.putText(image_np_with_detections, detected_text, 
                        (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (36,255,12), 2)
            
            detected_plate_text = detected_text  # Store the detected text

    return image_np_with_detections, detected_plate_text

def detect_fn(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections