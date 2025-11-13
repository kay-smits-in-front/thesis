"""Added configurations for GPU usage in machine learning models."""
import tensorflow as tf

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		print(f"GPU(s) detected: {len(gpus)}")
		print(f"GPU name: {gpus[0].name}")
	except RuntimeError as e:
		print(e)
else:
	print("No GPU detected - using CPU")
# TensorFlow-compatible MMG Physics Functions