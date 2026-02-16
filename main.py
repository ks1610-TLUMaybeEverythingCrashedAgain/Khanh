from ultralytics import YOLO
import cv2
import numpy as np
import math
import os

# --- CẤU HÌNH ---
MODEL_PATH = r'best.pt' # chọn best hoặc last tùy kết quả huấn luyện
IMAGE_PATH = r'test.jpg' 
CONFIDENCE_THRESHOLD = 0.4
OVERLAP_RATIO = 0.5  # Tăng lên 50% để đảm bảo không bỏ sót linh kiện ở mép cắt

# --- HÀM TÍNH KÍCH THƯỚC SCALE (Giữ nguyên logic của bạn) ---
def find_scale(size):
    size = int(size * 0.2) # 20% kích thước ảnh
    if size < 32:
        size = 32
    if size % 32 == 0:
        return size
    return ((size // 32) + 1) * 32

# --- 1. HÀM TIỀN XỬ LÝ ẢNH ---
def preprocess_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# --- 2. HÀM THÊM PADDING ---
def pad_image(image, tile_size, stride):
    img_h, img_w = image.shape[:2]
    
    # Tính toán phần dư cần thêm vào
    pad_h = (tile_size - (img_h % stride)) % stride
    pad_w = (tile_size - (img_w % stride)) % stride
    
    # Cộng thêm tile_size vào padding để đảm bảo quét hết biên
    # (Tùy chọn: có thể tăng padding nếu muốn quét kỹ hơn ở mép)
    pad_h += int(tile_size * 0.5) 
    pad_w += int(tile_size * 0.5)

    # Sử dụng cv2.copyMakeBorder để thêm viền đen (BORDER_CONSTANT)
    # Trả về ảnh đã pad và kích thước pad để sau này trừ ngược lại tọa độ
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    return padded_img

# --- 3. HÀM NHẬN DIỆN SLIDING WINDOW ---
def predict_sliding_window(model, source_img, tile_size, overlap=0.5, conf=0.5):
    # Tính bước nhảy (stride). Overlap càng cao thì stride càng nhỏ, quét càng kỹ.
    stride = int(tile_size * (1 - overlap))
    
    # 1. Thêm Padding cho ảnh gốc
    padded_img = pad_image(source_img, tile_size, stride)
    pad_h, pad_w = padded_img.shape[:2]
    
    all_boxes = []
    all_scores = []
    all_class_ids = []

    print(f"🔄 Đang xử lý Sliding Window...")
    print(f"   - Kích thước gốc: {source_img.shape[1]}x{source_img.shape[0]}")
    print(f"   - Kích thước sau padding: {pad_w}x{pad_h}")
    print(f"   - Tile Size: {tile_size} | Stride: {stride} | Overlap: {int(overlap*100)}%")

    # 2. Duyệt vòng lặp (Correlation-like traversal)
    # Duyệt y từ 0 đến hết chiều cao đã pad, bước nhảy là stride
    count = 0
    for y in range(0, pad_h - tile_size + 1, stride):
        for x in range(0, pad_w - tile_size + 1, stride):
            
            # Cắt ảnh (Crop)
            tile = padded_img[y:y+tile_size, x:x+tile_size]
            
            # Nếu tile cắt ra bị nhỏ hơn kích thước quy định (ở mép cuối), bỏ qua
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue

            count += 1
            # Nhận diện
            results = model.predict(tile, conf=conf, verbose=False)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    
                    # 3. Mapping tọa độ: Cộng thêm vị trí của ô cắt (x, y)
                    global_x1 = int(bx1 + x)
                    global_y1 = int(by1 + y)
                    global_x2 = int(bx2 + x)
                    global_y2 = int(by2 + y)
                    
                    # Kiểm tra: Nếu hộp nằm hoàn toàn trong vùng padding (ngoài ảnh gốc), bỏ qua
                    if global_x1 >= source_img.shape[1] or global_y1 >= source_img.shape[0]:
                        continue

                    # Giới hạn tọa độ trong khung ảnh gốc
                    global_x1 = min(max(0, global_x1), source_img.shape[1])
                    global_y1 = min(max(0, global_y1), source_img.shape[0])
                    global_x2 = min(max(0, global_x2), source_img.shape[1])
                    global_y2 = min(max(0, global_y2), source_img.shape[0])

                    all_boxes.append([global_x1, global_y1, global_x2 - global_x1, global_y2 - global_y1])
                    all_scores.append(float(box.conf[0]))
                    all_class_ids.append(int(box.cls[0]))
    
    print(f"✅ Đã quét xong {count} ô cửa sổ.")
    return all_boxes, all_scores, all_class_ids

# --- 4. HÀM MAIN ---
def detect_and_highlight():
    try:
        # Kiểm tra tồn tại
        if not os.path.exists(MODEL_PATH) or not os.path.exists(IMAGE_PATH):
            print("❌ Lỗi: Không tìm thấy file model hoặc ảnh.")
            return

        print(f"⏳ Đang tải model từ: {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)

        original_img = cv2.imread(IMAGE_PATH)
        if original_img is None:
            print("❌ Lỗi đọc ảnh.")
            return

        # Tính kích thước tile động
        whsize = find_scale(max(original_img.shape[:2]))
        
        # Tiền xử lý
        processed_img = preprocess_image(original_img)
        
        # Chạy Sliding Window
        boxes, scores, class_ids = predict_sliding_window(
            model, 
            processed_img, 
            tile_size=whsize, 
            overlap=OVERLAP_RATIO, 
            conf=CONFIDENCE_THRESHOLD
        )

        # NMS (Cực kỳ quan trọng khi overlap cao)
        # Tăng overlap dẫn đến 1 vật thể bị phát hiện nhiều lần -> NMS sẽ gộp lại
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=0.4)

        print(f"🎯 Kết quả: Tìm thấy {len(indices)} linh kiện duy nhất.")

        for i in indices:
            idx = i if isinstance(i, (int, np.integer)) else i[0]
            x, y, w, h = boxes[idx]
            label = str(model.names[class_ids[idx]])
            score = scores[idx]

            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(original_img, f"{label} {score:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Resize hiển thị
        display_img = original_img.copy()
        h, w = display_img.shape[:2]
        if w > 1280:
            scale = 1280 / w
            display_img = cv2.resize(display_img, (1280, int(h * scale)))

        cv2.imshow("Sliding Window Result", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("ket_qua.jpg", original_img)

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    detect_and_highlight()