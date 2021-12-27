import argparse
import time
from pathlib import Path
from datetime import datetime
from threading import Thread
from collections import deque
from torch.multiprocessing import Process, set_start_method
import cv2,os
import torch
import torch.backends.cudnn as cudnn
from skimage.metrics import structural_similarity
from interval import Interval
from numpy import random
import configparser
from abs import Abs
import matplotlib.path as mpltPath
import numpy as np
from os import walk
import psutil
from rtsp import multiThread
from rmbredaydir import rmBreday
from mark_points_on_image import MarkPoints

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from linkpost import *
import queue 
import pygame
import keyboard
import glob

def stop_to_save_video(currentStatus, startTime, anomalyCamID, threadInstance):
    '''
    中斷影片儲存
    '''
    threadInstance.terminate()
    time.sleep(1)
    if threadInstance.is_alive():
        print('live')
    else:
        print('close')
    time.sleep(0.1)    
    if currentStatus:
        status = "Alarm"
    else:
        status = "ok" 
    date = startTime.strftime('%Y.%m.%d') 
    startTime = startTime.strftime('%H_%M_%S')
    os.remove('saveVideo/{0}/{1}/{2}_CAM_{0}_{3}.avi'.format(anomalyCamID, date, startTime, status))

def save_video(domainIP, currentStatus, startTime, anomalyCamID, storage_sec, threadInstance): 
    '''
    開始影片儲存
    '''
    if currentStatus:
        status = "Alarm"
    else:
        status = "ok"   

    if currentStatus == 'Monitor':
        status = "Monitor"

    url = 'rtsp://{0}.{1}/h265'
    date = startTime.strftime('%Y.%m.%d')
    startTime = startTime.strftime('%H_%M_%S')
    subProcess = Thread(target=threadInstance.multithread_run, args=(currentStatus, anomalyCamID, storage_sec, url.format(domainIP, anomalyCamID), date, startTime, status))
    subProcess.start()
    return subProcess

def play_sound(anomalyCamID ,remind):
    '''
    播放警報音樂
    '''
    if remind:
        sound = 'sound/remind.mp3'
    else:
        if anomalyCamID < 88:
            sound = 'sound/Alarm_A.mp3'
        elif anomalyCamID < 91:
            sound = 'sound/Alarm_B.mp3'
        elif anomalyCamID < 94:
            sound = 'sound/Alarm_C.mp3'
        elif anomalyCamID < 96:
            sound = 'sound/Alarm_D.mp3'
    print('playsound')
    pygame.mixer.init()
    pygame.mixer.music.load(sound)
    pygame.mixer.music.play()

def mark_points_on_camera_view(img, viewName='viewName', numPoints=1):
    """set points on the camera view

    Args:
        img (img): camera view image
        viewName (str, optional): the view windows name. Defaults to 'viewName'.
        numPoints (int, optional): number of labels required . Defaults to 1.

    Returns:
        [type]: camera_view_points
    """    

    markPoints = MarkPoints(img, viewName)
    camera_view_points = markPoints.mark_points(numPoints)
    return camera_view_points

def SetIpCfg():
    SrvCfg = configparser.ConfigParser()
    SrvCfg.read('./cfg/Service.cfg')
    domainIP = SrvCfg.get('Threshold', 'domainIP')
    return domainIP

def SetSrvCfg(anomalyCamID):
    '''
    config資料撈取
    '''
    global storage_sec, margin, detect_sec, buffer, monitorNo ,monitor_sec, breday
    global normalImg, goodshight, zebratape, limit
    global goodshight_open, goodshight_close, others_open, others_close, remind_open, remind_close, rmbreday_open, rmbreday_close
    global subCateNo, systemCode
    SrvCfg = configparser.ConfigParser()
    SrvCfg.read('./cfg/Service.cfg')
    storage_sec = int(SrvCfg.get('Threshold', 'storage_sec'))
    monitor_sec = int(SrvCfg.get('Threshold', 'monitor_sec'))
    breday = int(SrvCfg.get('Threshold', 'breday'))
    normalImg = SrvCfg.get(str(anomalyCamID), 'normalImg')  
    goodshight = eval(SrvCfg.get(str(anomalyCamID), 'goodshight'))
    zebratape = eval(SrvCfg.get(str(anomalyCamID), 'zebratape'))
    limit = eval(SrvCfg.get(str(anomalyCamID), 'limit'))
    monitorNo = eval(SrvCfg.get('Threshold', 'monitorNo'))
    margin = int(SrvCfg.get('Threshold', 'margin'))
    detect_sec =  int(SrvCfg.get('Threshold', 'detect_sec'))
    buffer = float(SrvCfg.get('Threshold', 'buffer'))
    subCateNo = SrvCfg.get('Link', 'subCateNo')
    systemCode = SrvCfg.get('Link', 'systemCode')
    goodshight_open = SrvCfg.get('Goodshight', 'open')
    goodshight_close = SrvCfg.get('Goodshight', 'close')
    others_open = SrvCfg.get('Others', 'open')
    others_close = SrvCfg.get('Others', 'close')
    remind_open = SrvCfg.get('Remind', 'open')
    remind_close = SrvCfg.get('Remind', 'close')
    rmbreday_open = SrvCfg.get('Rmbreday', 'open')
    rmbreday_close = SrvCfg.get('Rmbreday', 'close')

def SetSrvCfgSampling():
    global domainIP, monitorNo, monitor_sec
    SrvCfg = configparser.ConfigParser()
    SrvCfg.read('./cfg/Service.cfg')
    domainIP = SrvCfg.get('Sampling', 'domainIP')
    monitorNo = eval(SrvCfg.get('Sampling', 'monitorNo'))

def parse_all_argument():
    '''
    變數設置
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='cfg/L3B.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/test_data', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--margin', type=int, default=0, help='threshold for margin')    
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    return opt

def get_limit_time(limit_open, limit_close):
    now_localtime = time.strftime("%H:%M:%S", time.localtime())
    now_time = Interval(now_localtime, now_localtime)
    limit_time = Interval(limit_open, limit_close)
    return now_time, limit_time

def end_of_detect(normalImg, start, fpsNum, camStage, camn, alarmNum, anomalyCamID, currentStatus, finalAnomalyTypes, finalAnomalyFrame, threadInstance): 
    end = datetime.now()
    if (end - start).seconds > detect_sec:
        if buffer == 0:         #檢測靈敏度控制(0:只要其中frame為alarm即為alarm ~ 1:每frame都為alarm才為alarm)
            threshold = fpsNum
        elif 1 >= buffer > 0:
            threshold = fpsNum / ( fpsNum * buffer )
        else:
            print("error: buffer > 1 or buffer < 0")    #buffer必須為0~1之間
            exit()

        camStage[camn][0] = camStage[camn][1]
        if  alarmNum != 0:
            if fpsNum / alarmNum  <=  threshold:
                camStage[camn][1] = True
                print('Alarm')
                if ((not camStage[camn][0]) and camStage[camn][1]):     # ok -> Alarm
                    now_time, limit_time = get_limit_time(remind_open, remind_close)
                    if now_time in limit_time:                #Remind提醒
                        play_sound(anomalyCamID, remind = True)
                    else:                                        #Alarm警告
                        play_sound(anomalyCamID, remind = False)
                    Linkpost(anomalyCamID, finalAnomalyTypes, finalAnomalyFrame, subCateNo, systemCode)
                elif(camStage[camn][0] and camStage[camn][1]):   # Alarm -> Alarm 
                    stop_to_save_video(currentStatus, start, anomalyCamID, threadInstance)
            else:
                camStage[camn][1] = False
                print('OK')
                if (camStage[camn][0] and (not camStage[camn][1])):     # Alarm -> ok 
                    Linkpost(anomalyCamID, "None", normalImg, subCateNo, systemCode)
                elif ((not camStage[camn][0]) and (not camStage[camn][1])):     # ok -> ok  
                    stop_to_save_video(currentStatus, start, anomalyCamID, threadInstance) 
        else:
            camStage[camn][1] = False
            print('OK')
            if (camStage[camn][0] and (not camStage[camn][1])):     # Alarm -> ok 
                Linkpost(anomalyCamID, "None", normalImg, subCateNo, systemCode)
            elif ((not camStage[camn][0]) and (not camStage[camn][1])):     # ok -> ok  
                stop_to_save_video(currentStatus, start, anomalyCamID, threadInstance)     

        cv2.destroyAllWindows()
        return True

def detect(save_img=False):
    '''
    檢測程式
    '''
    global camn, camStage
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 載入魚眼辨識權重
    caliArray = np.load(r'./cfg/R3V6F_720p_matrix.npz')

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    firstIP = 86 #初始IP
    camn = 0  
    source = []
    dataset = []
    camNum = 10 #IPcam數量
    camStage = np.full((camNum, 2), False)
    domainIP = SetIpCfg()
    for i in range(camNum):
        source.append('rtsp://{0}.{1}/h265'.format(domainIP, firstIP + i))

    webcam = True    
    # Set Dataloader
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        for i in source:
            dataset.append(LoadStreams(i, caliArray, img_size=imgsz))
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # initial =  False
    processQueue = []
    monitorQueue = []
    monitorThre = False
    while True:
        alarmNum = 0
        fpsNum = 0
        start = datetime.now()
        
        for queue in processQueue:
            if queue.is_alive():
                print(queue)

            if not queue.is_alive():
                    print("stop", queue)
        anomalyCamID = camn + firstIP
        SetSrvCfg(anomalyCamID)
        now_time, rmbreday_time = get_limit_time(rmbreday_open, rmbreday_close)
        if now_time in rmbreday_time:
            rmschedule = Thread(target=rmBreday, args = (breday,))
            rmschedule.start()
        for queue in monitorQueue:
            if not queue.is_alive():
                monitorThre = False
        if not monitorThre :
            monitorQueue = []
            for i in range(len(monitorNo)):
                threadInstance = multiThread()
                currentProcess = save_video(domainIP, 'Monitor', start, monitorNo[i], monitor_sec, threadInstance)
                monitorQueue.append(currentProcess)
                monitorThre = True

        ##################################################### 無限迴圈 不斷抓取 camera x 的影像   Start
        finalAnomalyTypes = None
        finalAnomalyFrame = None
        for path, img, im0s, vid_cap in dataset[camn]:

            if fpsNum == 0:
                currentStatus = not camStage[camn][1]
                threadInstance = multiThread()
                currentProcess = save_video(domainIP, currentStatus, start, anomalyCamID, storage_sec, threadInstance)
                processQueue.append(currentProcess)

            ##################################################### One  Frame   Start
            t1 = time_synchronized() 
            img = torch.from_numpy(img).to(device)          
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

           #####################################################  執行一次, det為所有boundingBox種類個數  in one Frame  Start
            for i, det in enumerate(pred):  # detections per image
                violation_frame=[]
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                goodshight_margin = [(goodshight[0][0], goodshight[0][1]), (goodshight[1][0], goodshight[1][1]), (goodshight[2][0], goodshight[2][1] - margin), (goodshight[3][0], goodshight[3][1] - margin)]
                zebratape_margin = [(zebratape[0][0], zebratape[0][1] - margin), (zebratape[1][0], zebratape[1][1] - margin), (zebratape[2][0], zebratape[2][1] + margin), (zebratape[3][0], zebratape[3][1] + margin)]
                limit_margin = [(limit[0][0] + margin, limit[0][1] - margin), (limit[1][0] - margin, limit[1][1] - margin), (limit[2][0] - margin, limit[2][1] + margin), (limit[3][0] + margin, limit[3][1] + margin)]
                abnormal = Abs(normalImg, im0, goodshight, zebratape, limit)
                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string

                ##################################################### All  boundingBox in one Frame   Start
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string

                    # Write results
                    anomalyTypes = []

                    for *xyxy, cls in reversed(det):

                        ##################################################### One boundingBox   Start
                        if save_img or view_img:  # Add bbox to image
                            label = names[int(cls)]
                            if label == 'YBox':
                                color=(0, 0, 255)
                            else:
                                color=(255, 0, 0)
                            
                            violation, anomalyType = plot_one_box(xyxy, im0, abnormal, goodshight_margin, zebratape_margin, limit_margin, goodshight_open, goodshight_close, others_open, others_close, label, color, line_thickness=2)
                            violation_frame.append(violation)
                            if violation == 'Alarm':
                                anomalyTypes.append(anomalyType)
                        ##################################################### One boundingBox   End
                        
                    if len(anomalyTypes) > 0:  ##### 代表一張Frame中有異常
                        alarmNum+=1
                        anomalyTypes = np.unique(anomalyTypes).tolist()
                        finalAnomalyTypes = anomalyTypes
                        finalAnomalyFrame = im0                
                ##################################################### All  boundingBox  in one Frame   End                
            #####################################################  執行一次 det為所有boundingBox種類個數 in one Frame   End           
            fpsNum+=1
            t2 = time_synchronized()
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            font = cv2.FONT_HERSHEY_SIMPLEX
            now_localtime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            cv2.putText(im0, 'FPS ' + str(round(1/(t2 - t1), 2)), (1000,50), font, 1.2, (255,0,0), 2)
            cv2.putText(im0, now_localtime, (50,50), font, 1.2, (255,0,0), 2)
            print(datetime.now())

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
            ##################################################### One  Frame   End
            ##################################################### 確定 camera x 已做完5秒才開始下判斷與影片存檔   Start         
            #影片檢測至detect_sec秒數，判斷此輪result為何
            result = False
            result = end_of_detect(im0, start, fpsNum, camStage, camn, alarmNum, anomalyCamID, currentStatus, finalAnomalyTypes, finalAnomalyFrame, threadInstance)
            if result : 
                if camn == 9:
                    camn = 0
                else:
                    camn += 1                
                break
        if keyboard.is_pressed("q"):
            break
        print(f'All Done. ({time.time() - t0:.3f}s)')
        ##################################################### 無限迴圈 不斷抓取 camera x 的影像   End 

def run():
    global opt
    opt = parse_all_argument()
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

def sampling(monitor_sec):
    SetSrvCfgSampling()
    monitorQueue = []
    monitorThre = False
    threadend = 0
    #time.sleep(0.1)        
    start = datetime.now()

    for queue in monitorQueue:
        if not queue.is_alive():
            threadend +=1

    if len(monitorQueue) ==  threadend:
        monitorThre = False
        threadend = 0

    if not monitorThre :
        monitorQueue = []
        for i in range(len(monitorNo)):
            threadInstance = multiThread()
            currentProcess = save_video(domainIP, 'Monitor', start, monitorNo[i], monitor_sec, threadInstance)
            monitorQueue.append(currentProcess)
        monitorThre = True

def write_to_cfg(config, no, prohibitedPoints, type):    
    config[no][type] = str(prohibitedPoints)
    with open('./cfg/Service_test.cfg', 'w') as configfile:
        config.write(configfile)

def bounfing(CAMIP):
    config = configparser.ConfigParser()
    config.read('./cfg/Service_test.cfg')
    image_dir = "./OKMaskScenestest"
    file_glob = os.path.join(image_dir, "*.png")
    file_list = []
    file_list.extend(glob.glob(file_glob))
    #for i in file_list:
    CAMIP = os.path.basename(i).strip('.png')
    img = cv2.imread(i)

    ### Prohibited Area: 1.over 2m (4 points)
    prohibitedPoints1 = mark_points_on_camera_view(img, "Prohibited Area: 1.over 2m (4 points)", 4)
    cv2.destroyAllWindows()
    print(f"prohibitedPoints1: {prohibitedPoints1}")
    write_to_cfg(config , CAMIP, prohibitedPoints1, 'goodshight')

    ### Prohibited Area: 2.zebra tape on the ground(4 points)
    prohibitedPoints2 = mark_points_on_camera_view(img, "Prohibited Area: 2.on the ground1(4 points)", 4)
    cv2.destroyAllWindows()
    print(f"prohibitedPoints2: {prohibitedPoints2}")
    write_to_cfg(config , CAMIP, prohibitedPoints2, 'zebratape')

    ### Prohibited Area: 3.aisle(4 points)
    prohibitedPoints3 = mark_points_on_camera_view(img, "Prohibited Area: 3.aisle(4 points)", 4)
    cv2.destroyAllWindows()
    print(f"prohibitedPoints3: {prohibitedPoints3}")
    write_to_cfg(config , CAMIP, prohibitedPoints3, 'limit')

def Initial_Image(CAMIP):
    SetSrvCfgSampling()
    url = 'rtsp://{0}.{1}/h265' 
    vcap = cv2.VideoCapture(url.format(domainIP, CAMIP))
    while(1): 
        try:
            ret, frame = vcap.read()
            cv2.putText(frame, f"Do you need to save the image? ",
                (1, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"- Press y key to save",
                                    (1, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"- Press n key to pass",
                                    (1, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('{}'.format(CAMIP), frame)

        except Exception as e:
            print("Error:" + str(e))
            vcap  = cv2.VideoCapture(url.format(domainIP, CAMIP))
            continue
        k=cv2.waitKey(1)
        # 按a拍照存檔
        if k & 0xFF == ord('y'):
            cv2.imwrite('./OKMaskScenestest/{}.png'.format(CAMIP), frame)
            break    
        # 按q離開
        if k & 0xFF == ord('n'):
            break
    cv2.destroyAllWindows()