import cv2
import numpy as np


def preprocess_image(color_image, model_input_size=(640, 640)):
    """
    Preprocess the input RGB color image for YOLO model including brightness equalization and noise reduction.

    Args:
    color_image (np.array): Input RGB color image as NumPy array.
    model_input_size (tuple): Size to which the image will be resized (default is 416x416).

    Returns:
    np.array: Preprocessed image.
    """
    # Cân bằng sáng (Histogram Equalization) cho từng kênh màu
    equalized_image = color_image.copy()
    for channel in range(3):  # RGB channels
        equalized_image[:,:,channel] = cv2.equalizeHist(color_image[:,:,channel])

    # Giảm nhiễu bằng Gaussian Blur
    denoised_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    # Thay đổi kích thước hình ảnh về kích thước đầu vào của mô hình
    resized_image = cv2.resize(denoised_image, model_input_size, interpolation=cv2.INTER_LINEAR)

    # Chuyển đổi hình ảnh từ BGR (mặc định của OpenCV) sang RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Chuẩn hóa giá trị pixel (đưa về khoảng [0, 1])
    normalized_image = rgb_image / 255.0

    # Chuyển đổi hình ảnh sang định dạng float32
    preprocessed_image = np.asarray(normalized_image, dtype=np.float32)

    # Thêm chiều batch (1, model_input_size[0], model_input_size[1], 3)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    return preprocessed_image
