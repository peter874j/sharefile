import cv2
import numpy as np
import os
import time


if __name__ == '__main__':
    ### Load Camera Calibration File
    caliFile = np.load(r'D:\_Project\E-fence\camera_relation\calibration_matrix\R3V6F\720p_R3V6F_best.npz')
    matrix = caliFile['mtx']
    distortion = caliFile['dist']
    videoPath = r'D:\_Project\E-fence\camera_relation\Cobot_E-Fence_video'
    outputPath = r'D:\_Project\E-fence\camera_relation\Output'
    ### iterates video floder
    for root, dirs, files in os.walk(videoPath):
        for f in files:
            ### Test Video
            videoFile = os.path.join(root, f)
            cap = cv2.VideoCapture(videoFile)    # from video file
            ### Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            outVideo = cv2.VideoWriter(videoFile.replace('Cobot_E-Fence_video', 'Output'), fourcc, 30.0, (1280,  720))
            ### captures stream
            while cap.isOpened():
                key = cv2.waitKey(1)
                ret, frame = cap.read()
                if ret:
                    img = frame.copy()
                    # cv2.imwrite('original.jpg', img)
                    imgH,  imgW = img.shape[:2]

                    ### Calibration Start
                    t1 = time.time()    
                    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (imgW, imgH), 1, (imgW, imgH))
                    ### undistort
                    dst = cv2.undistort(img, matrix, distortion, None, newCameraMatrix)
                    cv2.imwrite('undistort.jpg', dst)
                    ### Calibration End

                    ### Crops Images Start
                    cropX, cropY, cropW, cropH = roi
                    dst = dst[cropY:cropY + cropH, cropX:cropX + cropW]
                    t2 = time.time()    # Calibration Crop End
                    cv2.imwrite('calibration.jpg', dst)
                    ### Crops Images End

                    ### Resize Start
                    dst = cv2.resize(dst, (imgW, imgH), interpolation = cv2.INTER_AREA)
                    t3 = time.time()    # Calibration Crop&Resize End
                    # print('Crop Calibration Time:' + str(t2 - t1))
                    # print('Crop & Resize Calibration Time:' + str(t3 - t1))
                    # cv2.imwrite('resize.jpg', dst)
                    ### Resize End

                    ### write the frame
                    outVideo.write(dst)
                    # cv2.imshow('frame', img)
                    # cv2.waitKey(250)
                else:
                    break
            ### Release everything if job is finished
            cap.release()
            cv2.destroyAllWindows()


