from threading import Thread
from collections import deque
from multiprocessing import Process
import cv2
import time
from datetime import datetime
import os
import keyboard

class multiThread(Thread):

	def __init__(self):
		super().__init__()
		self._running = True

	def terminate(self):
		self._running = False

	def producer(self, cap, q, camera_index, storage_sec, f, write_fps):
		frame = 0
		while self._running:
			print('captures image...')
			try:
				ret = cap.grab()
				if keyboard.is_pressed("q"):
					break
				
				if not ret:
					print("camera {} error".format(camera_index))
				else:
					frame += 1
				if frame % (15 / write_fps) == 0:
					ret, img = cap.retrieve()
					font = cv2.FONT_HERSHEY_SIMPLEX
					now_localtime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
					cv2.putText(img, now_localtime, (50,50), font, 1.2, (255,0,0), 2)
					q.append(img)
					f.append(frame)
				if frame >= storage_sec * 15 :
					break
			except:
				print("camera {} error".format(camera_index))

	def consumer(self, camera_index, storage_sec, outVideo, q, f):
		print("Start to capture and save video of camera {}...".format(camera_index))
		while self._running:
			print('records frame...')
			try:
				if keyboard.is_pressed("q"):
					break

				if len(q) == 0:
					pass
				else:
					img = q.pop()
					frame = f.pop()
					outVideo.write(img)
					if frame >= storage_sec * 15 :
						break
			except:
				print("camera {} error".format(camera_index))

	def multithread_run(self, currentStatus, anomalyCamID, storage_sec, url, date, startTime, status):
		start = datetime.now()
		width =  1280
		height = 720
		fps = 15
		fourcc = cv2.VideoWriter_fourcc('M','P','4','2')
		if currentStatus == 'Monitor':
			if not os.path.isdir('monitor/{0}/{1}'.format(anomalyCamID, date)): os.mkdir('monitor/{0}/{1}'.format(anomalyCamID, date))
			outVideo = cv2.VideoWriter('monitor/{0}/{1}/{2}_CAM_{0}_{3}.avi'.format(anomalyCamID, date, startTime, status), fourcc, fps, (width, height))
			write_fps = 15
		else:
			if not os.path.isdir('saveVideo/{0}/{1}'.format(anomalyCamID, date)): os.mkdir('saveVideo/{0}/{1}'.format(anomalyCamID, date))
			outVideo = cv2.VideoWriter('saveVideo/{0}/{1}/{2}_CAM_{0}_{3}.avi'.format(anomalyCamID, date, startTime, status), fourcc, fps, (width, height))
			write_fps = 3
		q = deque(maxlen=1)
		f = deque(maxlen=1)
		cap = cv2.VideoCapture(url)	
		width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		print("Frame Width = {}, Frame Height = {}".format(width, height))
		p1 = Thread(target=self.producer, args=(cap, q, anomalyCamID, storage_sec, f, write_fps))
		c1 = Thread(target=self.consumer, args=(anomalyCamID, storage_sec, outVideo, q, f))
		p1.start()  
		c1.start()
		p1.join()
		c1.join()
		end = datetime.now()
		print("execution time: ", end - start)