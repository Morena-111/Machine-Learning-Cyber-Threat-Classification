import pickle
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from src.data_loader import load_datasets
from src.preprocess import extract_entity_examples, prepare_text_data
from src.model import build_model

# 1. Load model & vocab
with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('models/tokenizer.pkl', 'rb') as f:
    vocab = pickle.load(f)

v = tf.keras.layers.TextVectorization(
    max_tokens=10000, 
    output_mode='int', 
    output_sequence_length=50, 
    standardize='lower_and_strip_punctuation'
)
v.set_vocabulary(vocab)
model = build_model(v, num_classes=len(encoder.classes_))
model.load_weights('models/best_model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 2. Load 1 test example
_, _, raw_test = load_datasets('data')
test_examples = extract_entity_examples(raw_test)
texts = prepare_text_data(test_examples[:1]) # Just 1 example for the bar chart

# 3. Generate SHAP
print('Generating SHAP values... (this takes ~30 seconds)')
explainer = shap.Explainer(
    lambda x: model.predict(tf.constant(x, dtype=tf.string), verbose=0), 
    shap.maskers.Text()
)
shap_values = explainer(texts)

# 4. Plot the top 10 features
plt.figure(figsize=(10, 5))
shap.plots.bar(shap_values[0, :, 0], max_display=10, show=False)
plt.title("Top features driving the malware prediction", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/shap_explanation.png', dpi=300, bbox_inches='tight')
print('Saved clean SHAP plot to results/shap_explanation.png')
