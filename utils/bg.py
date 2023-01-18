import numpy as np
import cv2

class bg:
    def __init__(self, fg_path):
        self.path = fg_path
        self.fg = cv2.VideoCapture(self.path)
        self.fg_length = int(self.fg.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.fg_length)
        
    def run(self, frame, level):
        print(int(self.fg.get(cv2.CAP_PROP_POS_FRAMES)))
        if self.fg_length == int(self.fg.get(cv2.CAP_PROP_POS_FRAMES)):
            print('reset')
            self.fg.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, foreground = self.fg.read()
        # foreground = cv2.resize(foreground, (640,480), interpolation = cv2.INTER_AREA)
        
        # creating the alpha mask
        alpha = np.zeros_like(foreground)
        gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        alpha[:, :, 0] = gray
        alpha[:, :, 1] = gray
        alpha[:, :, 2] = gray

        #converting uint8 to float type
        foreground = foreground.astype(float)
        frame = frame.astype(float)

        # normalizing the alpha mask inorder
        # to keep intensity between 0 and 1
        alpha = alpha.astype(float)/255*level

        # multiplying the forground
        # with alpha matte
        foreground = cv2.multiply(alpha, foreground)

        # multiplying the origin frame
        # with (1 - alpha)
        frame = cv2.multiply(1.0 - alpha, frame)

        # adding the masked foreground
        # and origin frame together
        outImage = cv2.add(foreground, frame)

        return outImage


