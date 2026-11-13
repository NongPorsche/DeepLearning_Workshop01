import os
import cv2
from ultralytics import YOLO

# 1. โหลดโมเดล
model = YOLO("Result_Model/Train1.pt")

# 2. กำหนดโฟลเดอร์รูปต้นทาง และโฟลเดอร์สำหรับเซฟผลลัพธ์
input_folder = "Pictures_for_Test"
output_folder = "Output_Pictures"
os.makedirs(output_folder, exist_ok=True)

# นามสกุลไฟล์รูปภาพที่รองรับ
valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
image_files = [
    f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)
]

print(f"พบรูปภาพทั้งหมด {len(image_files)} รูป กำลังเริ่มประมวลผล...")

for idx, filename in enumerate(image_files, 1):
    img_path = os.path.join(input_folder, filename)

    # สั่งโมเดลตรวจจับ
    results = model.predict(source=img_path, conf=0.2, verbose=False)

    # วาดกรอบลงบนภาพ
    annotated_img = results[0].plot()

    # เซฟรูปผลลัพธ์ลงโฟลเดอร์ output_results
    save_path = os.path.join(output_folder, f"result_{filename}")
    cv2.imwrite(save_path, annotated_img)

    # ดึงจำนวนวัตถุที่เจอ
    detections_count = len(results[0].boxes)
    print(
        f"[{idx}/{len(image_files)}] {filename} -> พบ Apple Pencil: {detections_count} จุด (บันทึกที่ {save_path})"
    )

    # แสดงผลทีละรูป (กดปุ่มใดๆ บนคีย์บอร์ดเพื่อดูรูปถัดไป หรือกด 'q' เพื่อหยุดดู)
    cv2.imshow("Detection Viewer", annotated_img)
    key = cv2.waitKey(0)
    if key & 0xFF == ord("q"):
        print("หยุดการแสดงผลหน้าต่างภาพ")
        break

cv2.destroyAllWindows()
print(f"ประมวลผลเสร็จสิ้น ตรวจสอบภาพผลลัพธ์ทั้งหมดได้ในโฟลเดอร์: {output_folder}")