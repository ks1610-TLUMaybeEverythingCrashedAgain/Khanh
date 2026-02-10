from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_scale(size):
    size = int(size * 0.2)
    if size < 32:
        size = 32

    if size % 32 == 0:
        return size

    return ((size // 32) + 1) * 32

# --- CẤU HÌNH ---
MODEL_PATH = r'best.pt'
IMAGE_PATH = r'test.jpg' #thay đường dẫn
CONFIDENCE_THRESHOLD = 0.4  
whsize = find_scale(max(cv2.imread(IMAGE_PATH).shape[:2]))
SLICE_SIZE = whsize         
OVERLAP_RATIO = 0.25        

# --- 1. HÀM TIỀN XỬ LÝ ẢNH (Pre-processing) ---
def preprocess_image(image):
    """
    Sử dụng CLAHE để cân bằng sáng cục bộ, giúp linh kiện nổi bật hơn trên nền mạch.
    """
    # Chuyển sang hệ màu LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Áp dụng CLAHE lên kênh L (Lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Gộp lại và chuyển về BGR
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# --- 2. HÀM NHẬN DIỆN CẮT LÁT (Tiled Inference) ---
def predict_tiled(model, source_img, tile_size=640, overlap=0.25, conf=0.5):
    img_h, img_w = source_img.shape[:2]
    
    # Tính bước nhảy (stride) dựa trên overlap
    stride = int(tile_size * (1 - overlap))
    
    all_boxes = []
    all_scores = []
    all_class_ids = []

    print(f"🔄 Đang xử lý ảnh kích thước {img_w}x{img_h} với ô cắt {tile_size}x{tile_size}...")

    # Duyệt qua từng ô của ảnh
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            # Xác định tọa độ cắt, đảm bảo không vượt quá kích thước ảnh
            x_end = min(x + tile_size, img_w)
            y_end = min(y + tile_size, img_h)
            x_start = x_end - tile_size if x_end - tile_size > 0 else 0
            y_start = y_end - tile_size if y_end - tile_size > 0 else 0

            # Cắt ảnh
            tile = source_img[y_start:y_end, x_start:x_end]

            # Nhận diện trên từng ô nhỏ
            results = model.predict(tile, conf=conf, verbose=False)
            
            # Xử lý kết quả
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Lấy tọa độ tương đối trong ô (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Chuyển đổi sang tọa độ toàn cục của ảnh gốc
                    global_x1 = int(x1 + x_start)
                    global_y1 = int(y1 + y_start)
                    global_x2 = int(x2 + x_start)
                    global_y2 = int(y2 + y_start)
                    
                    all_boxes.append([global_x1, global_y1, global_x2 - global_x1, global_y2 - global_y1]) # Format cho NMS: [x, y, w, h]
                    all_scores.append(float(box.conf[0]))
                    all_class_ids.append(int(box.cls[0]))

    return all_boxes, all_scores, all_class_ids

# --- 3. HÀM MAIN ---
def detect_and_highlight():
    try:
        # 1. Load Model
        print(f"Đang tải model từ: {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)

        # 2. Đọc ảnh
        original_img = cv2.imread(IMAGE_PATH)
        if original_img is None:
            print("❌ Không tìm thấy ảnh!")
            return

        # 3. Tiền xử lý ảnh (Làm nét, cân bằng sáng)
        processed_img = preprocess_image(original_img)
        
        # 4. Chạy nhận diện theo phương pháp cắt lát (Tiling)
        boxes, scores, class_ids = predict_tiled(
            model, 
            processed_img, 
            tile_size=SLICE_SIZE, 
            overlap=OVERLAP_RATIO, 
            conf=CONFIDENCE_THRESHOLD
        )

        # 5. Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các khung trùng nhau do cắt chồng lấn
        # 0.4 là ngưỡng giao nhau (IOU threshold)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=0.4)

        print(f"✅ Đã tìm thấy {len(indices)} linh kiện sau khi gộp kết quả.")

        # 6. Vẽ kết quả lên ảnh gốc
        for i in indices:
            # cv2.dnn.NMSBoxes trả về index dạng list hoặc mảng con, cần xử lý để lấy int
            idx = i if isinstance(i, (int, np.integer)) else i[0]
            
            x, y, w, h = boxes[idx]
            label = str(model.names[class_ids[idx]])
            score = scores[idx]

            # Vẽ khung chữ nhật
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Viết tên linh kiện
            cv2.putText(original_img, f"{label} {score:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"- {label}: {score:.2f}")

        # 7. Hiển thị kết quả
        # Resize ảnh nhỏ lại để hiển thị vừa màn hình nếu ảnh quá lớn
        display_img = original_img.copy()
        h, w = display_img.shape[:2]
        if w > 1280:
            scale = 1280 / w
            display_img = cv2.resize(display_img, (1280, int(h * scale)))

        cv2.imshow("Ket qua Nhan dien Nang cao (Tiled)", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Lưu ảnh full HD
        cv2.imwrite("ket_qua.jpg", original_img)
        print("💾 Đã lưu ảnh kết quả thành 'ket_qua_pcb_tiled.jpg'")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    detect_and_highlight()