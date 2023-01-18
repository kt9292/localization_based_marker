from utils.bg import bg
import cv2

fg_path = f'./bg/2.mp4'
fg = bg(fg_path)

bg = cv2.VideoCapture(0)

while True:
    _, frame = bg.read()
    result = fg.run(frame)

    cv2.imshow('test', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break