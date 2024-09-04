import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
import os

# Model paths and config
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PATHS = {
    'CHECKPOINT_PATH': os.path.join('model', 'checkpoints'),
}
FILES = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
}

def load_model():
    """Load the object detection model."""
    configs = config_util.get_configs_from_pipeline_file(FILES['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATHS['CHECKPOINT_PATH'], 'ckpt-61')).expect_partial()
    return detection_model
