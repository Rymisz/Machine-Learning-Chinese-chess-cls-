import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- 1. 全局参数和路径定义 ---
ALL_DATA_DIR = r"C:\Users\Rymis\Desktop\a\all_data" 

# 数据划分比例
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.15

# 模型和图像参数
IMG_HEIGHT = 96
IMG_WIDTH = 96
CHANNELS = 3
BATCH_SIZE = 32

# 训练周期与学习率 
EPOCHS = 100
INITIAL_LEARNING_RATE = 0.001 # 初始学习率

# --- 2. 自动化数据集划分与加载---
print("--- Step 1: Loading and Splitting Data (Robust Method) ---")
if not os.path.exists(ALL_DATA_DIR):
    print(f"Error: Data directory not found at {ALL_DATA_DIR}")
    exit()

temp_full_dataset = tf.keras.utils.image_dataset_from_directory(
    ALL_DATA_DIR, labels='inferred', image_size=(IMG_HEIGHT, IMG_WIDTH), shuffle=False
)
class_names = temp_full_dataset.class_names
NUM_CLASSES = len(class_names)
del temp_full_dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    ALL_DATA_DIR,
    validation_split=(VALIDATION_SPLIT + TEST_SPLIT),
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

temp_val_dataset = tf.keras.utils.image_dataset_from_directory(
    ALL_DATA_DIR,
    validation_split=(VALIDATION_SPLIT + TEST_SPLIT),
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_batches_to_take = int((TEST_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)) * len(temp_val_dataset))
test_dataset = temp_val_dataset.take(test_batches_to_take)
validation_dataset = temp_val_dataset.skip(test_batches_to_take)

print(f"Found {NUM_CLASSES} classes: {class_names}")
print(f"Training batches: {tf.data.experimental.cardinality(train_dataset)}")
print(f"Validation batches: {tf.data.experimental.cardinality(validation_dataset)}")
print(f"Test batches: {tf.data.experimental.cardinality(test_dataset)}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# --- 3. 模型构建 ---
print("\n--- Step 2: Building the effective Custom CNN Model with Enhanced Regularization ---")
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1), # 旋转幅度
        layers.RandomZoom(0.1),   # 缩放幅度
    ],
    name="data_augmentation",
)

def build_final_cnn_model(input_shape, num_classes):
    """构建最终优化的自定义CNN模型"""
    l2_reg = 0.005 # L2正则化因子

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Rescaling(1./255),

        layers.Conv2D(32, (3, 3), padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu',kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # Dropout率
        layers.Dense(num_classes, activation='softmax', name="output_layer")
    ], name="final_custom_cnn")
    
    return model

model = build_final_cnn_model((IMG_HEIGHT, IMG_WIDTH, CHANNELS), NUM_CLASSES)

# --- 4. 训练阶段 ---
print("\n--- Step 3: Training the model ---")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('final_best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6, verbose=1)
]

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=callbacks
)

# --- 5. 绘图与最终评估 ---
def plot_training_history(history):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    if not acc or not val_acc or not loss or not val_loss:
        print("Could not plot history: one of the metrics is empty.")
        return

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.savefig('final_training_curves.png')
    plt.show()

plot_training_history(history)

print("\nLoading best model for evaluation...")
if os.path.exists('final_best_model.keras'):
    best_model = keras.models.load_model('final_best_model.keras')
else:
    print("Warning: 'final_best_model.keras' not found. Evaluating with the last state of the model.")
    best_model = model

print("\nEvaluating final model on test set...")
test_loss, test_accuracy = best_model.evaluate(test_dataset)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Loss: {test_loss:.4f}')

print("\nGenerating detailed classification report and confusion matrix...")
y_pred_probs = best_model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true_one_hot = np.concatenate([y for x, y in test_dataset], axis=0)
y_true = np.argmax(y_true_one_hot, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 10})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('final_confusion_matrix.png')
plt.show()

print("\nScript finished successfully.")
