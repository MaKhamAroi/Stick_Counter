import cv2
import numpy as np

def detect_sticks(image_path):
    # อ่านภาพจากไฟล์
    img = cv2.imread(image_path)
    output = img.copy()
    
    # แปลงภาพเป็นขาวดำ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ลด noise ด้วย median blur
    blur = cv2.medianBlur(gray, 5)

    # เพิ่ม contrast ด้วย CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(blur)

    # ทำ sharpening เพื่อเน้นขอบ
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    # ลด noise เพิ่มเติมด้วย Gaussian blur
    final_blur = cv2.GaussianBlur(sharpened, (3, 3), 0)

    # ใช้ morphology เปิด เพื่อลด noise เล็กๆ
    morph = cv2.morphologyEx(final_blur, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    # ตรวจจับวงกลมด้วย HoughCircles
    circles = cv2.HoughCircles(
        morph,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=40,
        param2=23,
        minRadius=8,
        maxRadius=24
    )
    
    count = 0
    filtered_circles = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for circle in circles[0, :]:
            x, y, r = circle
            area = np.pi * (r ** 2)

            # กรองวงกลมด้วยพื้นที่ เพื่อกัน noise
            if area < 80 or area > 2000:
                continue

            is_duplicate = False
            for existing in filtered_circles:
                ex, ey, er = existing
                distance = np.sqrt((x - ex)**2 + (y - ey)**2)
                
                # ถ้าวงกลมอยู่ใกล้กันและขนาดใกล้เคียง ให้ถือว่าเป็นวงเดียวกัน
                if distance < (r + er) * 0.5 and abs(r - er) < 5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_circles.append(circle)
        
        count = len(filtered_circles)
        
        # วาดวงกลมที่ตรวจจับได้ลงบนภาพ
        for i in filtered_circles:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)  # วาดขอบวงกลม
            cv2.circle(output, (i[0], i[1]), 2, (255, 0, 0), 3)     # วาดจุดศูนย์กลาง
    
    # สร้างข้อความแสดงจำนวนวงกลมที่ตรวจจับได้
    text = f'Count: {count}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    # คำนวณขนาดของข้อความ
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # วาดพื้นหลังข้อความ
    cv2.rectangle(output, (5, 5), (text_width + 15, text_height + 15), (255, 255, 255), -1)
    # วาดกรอบข้อความ
    cv2.rectangle(output, (5, 5), (text_width + 15, text_height + 15), (0, 0, 0), 2)
    # วาดข้อความลงบนภาพ
    cv2.putText(output, text, (10, text_height + 8), font, font_scale, (0, 0, 255), thickness)
    
    return output, count
