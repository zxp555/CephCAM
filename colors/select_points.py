import cv2
import numpy as np

def select_points(img):
    # 初始化一个空列表，用于存储点的坐标
    points = []

    # 鼠标回调函数，用于记录点击的坐标
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键点击时记录坐标
            if len(points) < 4:
                # 将点击坐标转换到原始图像上的坐标
                orig_x = int((x - start_x) / scale)
                orig_y = int((y - start_y) / scale)
                
                # 防止坐标超出原始图像边界
                orig_x = np.clip(orig_x, 0, img_w - 1)
                orig_y = np.clip(orig_y, 0, img_h - 1)
                
                points.append((orig_x, orig_y))
                # print(f"Point {len(points)}: ({orig_x}, {orig_y})")
                # 在图像上画出点击的点
                cv2.circle(img_display, (x, y), 2, (0, 255, 0), -1)
                cv2.imshow('Image', img_display)

    # 读取图像
    # img = cv2.imread('path_to_your_image.jpg')  # 替换为你的图片路径
    img_h, img_w = img.shape[:2]

    # 创建一个可调整大小的窗口
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 获取最大化后的窗口大小
    cv2.imshow('Image', img)
    cv2.waitKey(10)  # 等待短暂时间以确保窗口最大化
    window_w, window_h = cv2.getWindowImageRect('Image')[2:]

    # 计算调整后的图像大小以适应窗口
    scale = min(window_w / img_w, window_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # 调整图像大小
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建一个黑色背景的画布来放置调整后的图像，使其居中
    img_display = 255 * np.ones((window_h, window_w, 3), dtype=np.uint8)
    start_x = (window_w - new_w) // 2
    start_y = (window_h - new_h) // 2
    img_display[start_y:start_y + new_h, start_x:start_x + new_w] = img_resized

    # 显示图像
    cv2.imshow('Image', img_display)

    # 设置鼠标回调函数
    cv2.setMouseCallback('Image', mouse_callback)

    # 等待用户设置4个点
    while True:
        if len(points) == 4:
            break
        # 按下 'q' 键可以提前退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()

    if len(points) == 4:
        return points
    else:
        return None

if __name__ == '__main__':
    print(select_points(cv2.imread('./wrapped.png')))