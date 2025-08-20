import tensorflow as tf

IMAGES = "/Users/raunaksmac/Desktop/Nutrition AI/food-6/images"  # mac
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32  # Reduced from 64 to help with stability
TOTAL_CLASSES = 6
FREEZE_EPOCHS = 5  # Reduced for better convergence
UNFREEZE_EPOCHS = 10
SEED = 123

train_data = tf.keras.utils.image_dataset_from_directory(
    IMAGES,
    validation_split=0.2,  # Using 20% for better validation
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_data = tf.keras.utils.image_dataset_from_directory(
    IMAGES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)


# Add this debugging code after loading your datasets to diagnose the issues

# 1. Check dataset size and class distribution
def analyze_dataset(dataset, name):
    print(f"\n=== {name} Dataset Analysis ===")
    
    # Count total samples
    total_samples = 0
    class_counts = {}
    
    for images, labels in dataset.unbatch():
        total_samples += 1
        # Convert one-hot to class index
        class_idx = tf.argmax(labels).numpy()
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
    
    print(f"Total samples: {total_samples}")
    print(f"Samples per class: {class_counts}")
    print(f"Classes found: {len(class_counts)}")
    
    # Check for severe class imbalance
    if class_counts:
        min_samples = min(class_counts.values())
        max_samples = max(class_counts.values())
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 3:
            print("⚠️  WARNING: Severe class imbalance detected!")
    
    return total_samples, class_counts

# Analyze your datasets
train_samples, train_class_counts = analyze_dataset(train_data, "Training")
val_samples, val_class_counts = analyze_dataset(val_data, "Validation")

# 2. Check if you have enough data
print(f"\n=== Data Sufficiency Check ===")
print(f"Training samples per class: {train_samples / TOTAL_CLASSES:.1f}")
print(f"Validation samples per class: {val_samples / TOTAL_CLASSES:.1f}")

if train_samples < 100 * TOTAL_CLASSES:
    print("⚠️  WARNING: Very small dataset! Consider data augmentation or reducing model complexity")

# 3. Verify data loading and preprocessing
print(f"\n=== Data Pipeline Check ===")
sample_batch = next(iter(train_data.take(1)))
images, labels = sample_batch
print(f"Batch shape: {images.shape}")
print(f"Image value range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
print(f"Labels shape: {labels.shape}")
print(f"Sample label distribution in batch: {tf.reduce_sum(labels, axis=0)}")

# Check if normalization is working correctly
if tf.reduce_max(images) > 1.1:
    print("⚠️  WARNING: Images might not be properly normalized!")

# 4. Test model prediction on a small batch
print(f"\n=== Model Prediction Test ===")
test_predictions = model.predict(images)
predicted_classes = tf.argmax(test_predictions, axis=1)
actual_classes = tf.argmax(labels, axis=1)

print(f"Predicted classes: {predicted_classes}")
print(f"Actual classes: {actual_classes}")
print(f"Prediction confidence (max prob): {tf.reduce_max(test_predictions, axis=1)}")

# Check if model is outputting reasonable probabilities
avg_max_confidence = tf.reduce_mean(tf.reduce_max(test_predictions, axis=1))
print(f"Average max confidence: {avg_max_confidence:.3f}")

if avg_max_confidence < 0.2:
    print("⚠️  WARNING: Model has very low confidence - might be outputting near-uniform probabilities")