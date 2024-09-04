import os

# Configuration for the model
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PATHS = {
    'CHECKPOINT_PATH': os.path.join('model', 'checkpoints'),
}
FILES = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
}

# Other settings can be added here
