import cv2
import numpy as np
from pyzbar import pyzbar
import zxingcpp

def decode_barcode(frame, frame_copy):
    barcodes = pyzbar.decode(frame, symbols=[
        pyzbar.ZBarSymbol.UPCE, pyzbar.ZBarSymbol.EAN13, pyzbar.ZBarSymbol.EAN8,
        pyzbar.ZBarSymbol.UPCA, pyzbar.ZBarSymbol.CODE128, pyzbar.ZBarSymbol.CODE39,
        pyzbar.ZBarSymbol.QRCODE,
    ])
    for barcode in barcodes:
        x, y, w, h = barcode.rect
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f'{barcode_data} ({barcode_type})'
        cv2.putText(frame_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame_copy

def detect_aruco(frame, frame_copy, dictionary=cv2.aruco.DICT_4X4_100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Dummy camera matrix and distortion coefficients for demonstration purposes
    camera_matrix = np.array([[1000, 0, frame.shape[1] / 2],
                              [0, 1000, frame.shape[0] / 2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

    if ids is not None:
        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame_copy, corners, ids)
            retval, rvecs, tvecs = cv2.solvePnP(
                np.array([[0, 0, 0], [0.04, 0, 0], [0.04, 0.04, 0], [0, 0.04, 0]], dtype=np.float32),
                corners[i][0],
                camera_matrix,
                dist_coeffs
            )
            cv2.drawFrameAxes(frame_copy, camera_matrix, dist_coeffs, rvecs, tvecs, 0.02)
    return frame_copy

def zxing_detect(frame, frame_copy, barcode_type):
    results = zxingcpp.read_barcodes(frame)
    for result in results:
        if result.format == barcode_type:
            x1, y1 = map(int, str(result.position).split()[0].split('x'))
            x2, y2 = map(int, str(result.position).split()[2].split('x'))
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{result.text} ({result.format})'
            cv2.putText(frame_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame_copy

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (720, 1280))
        
        # Make a copy of the frame to display the original frame later
        frame_copy = frame.copy()

        # Darken the frame for better detection
        frame = cv2.addWeighted(frame, 0.2, np.zeros(frame.shape, frame.dtype), 0, 0)
        frame_copy = cv2.addWeighted(frame_copy, 0.8, np.zeros(frame.shape, frame.dtype), 0, 0)

        frame_copy = decode_barcode(frame, frame_copy)
        frame_copy = detect_aruco(frame, frame_copy)
        frame_copy = detect_aruco(frame, frame_copy, cv2.aruco.DICT_APRILTAG_16H5)
        frame_copy = zxing_detect(frame, frame_copy, zxingcpp.BarcodeFormat.PDF417)
        frame_copy = zxing_detect(frame, frame_copy, zxingcpp.BarcodeFormat.MicroQRCode)
        frame_copy = zxing_detect(frame, frame_copy, zxingcpp.BarcodeFormat.DataMatrix)

        cv2.imshow('Barcode/QR code scanner', frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'barcodes.mp4'  # 비디오 파일 경로 설정
    main(video_path)
