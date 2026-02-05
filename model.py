import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from IPython.display import FileLink

# --- 1. CẤU HÌNH & LỌC DỮ LIỆU ---
DATA_DIR = '/kaggle/input/electronic-components/images'
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# Lấy tất cả tên folder và lọc bỏ folder 'images'
all_items = os.listdir(DATA_DIR)
valid_classes = [name for name in all_items 
                 if os.path.isdir(os.path.join(DATA_DIR, name)) 
                 and name != 'images']

valid_classes = sorted(valid_classes)
print(f"Đã tìm thấy {len(valid_classes)} loại linh kiện sạch.")
print(valid_classes[:5], "...")

# --- 2. LOAD DỮ LIỆU ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=valid_classes,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=valid_classes,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("\n--- DANH SÁCH NHÃN CHÍNH THỨC ---")
print(class_names)

# Tối ưu hiệu năng
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. XÂY DỰNG MÔ HÌNH ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

inputs = Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. HUẤN LUYỆN (PHASE 1) ---
print("\nBắt đầu huấn luyện Phase 1...")
# [SỬA ĐỔI] Giảm xuống 20 epochs là đủ để Transfer Learning hội tụ. 
# 150 epochs là quá thừa và tốn thời gian.
initial_epochs = 100
history = model.fit(train_ds, epochs=initial_epochs, validation_data=val_ds)

# --- 5. FINE-TUNING (PHASE 2) ---
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nBắt đầu Fine-tuning (Phase 2)...")
fine_tune_epochs = 10
# [QUAN TRỌNG] Tổng số epoch = epoch cũ + epoch mới
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_ds, 
                         epochs=total_epochs, # Phải set là 30 (20+10)
                         initial_epoch=history.epoch[-1], 
                         validation_data=val_ds)

# --- 6. LƯU MÔ HÌNH & DỮ LIỆU ---
# [FIX] Đưa phần tạo saved_data xuống đây (khi model đã có dữ liệu)
saved_data = {
    "model_weights": model.get_weights(),   
    "model_config": model.get_config(),     
    "class_names": class_names,             
    "img_size": IMG_SIZE                    
}

# [FIX] Định nghĩa tên file trước khi dùng
output_filename = "electronic_model_full.pickle"

print(f"\nĐang lưu dữ liệu vào {output_filename}...")
try:
    with open(output_filename, 'wb') as f:
        pickle.dump(saved_data, f)
    print("✅ Đã lưu file pickle thành công!")
except Exception as e:
    print(f"❌ Lỗi khi lưu file: {e}")

# Lưu file chuẩn .h5 và nhãn riêng
model.save('final_model.h5')
with open('labels.pickle', 'wb') as f:
    pickle.dump(class_names, f)

print("✅ Đã lưu: final_model.h5 và labels.pickle")

# Tạo link tải về
print("\n--- LINK TẢI VỀ ---")
display(FileLink(output_filename))
display(FileLink('final_model.h5'))
display(FileLink('labels.pickle'))

# --- 7. TEST DỰ ĐOÁN ---
def predict_random_sample():
    # Chọn ngẫu nhiên
    random_class = random.choice(class_names)
    class_path = os.path.join(DATA_DIR, random_class)
    
    # Kiểm tra folder rỗng
    if not os.listdir(class_path):
        print(f"Folder {random_class} rỗng, bỏ qua.")
        return

    random_image_name = random.choice(os.listdir(class_path))
    img_path = os.path.join(class_path, random_image_name)

    # Dự đoán
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0]
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"--- KẾT QUẢ TEST ---")
    print(f"Ảnh gốc: {random_class}")
    print(f"Dự đoán: {predicted_class} ({confidence:.2f}%)")
    
    plt.imshow(img)
    plt.axis("off")
    plt.show()

predict_random_sample()