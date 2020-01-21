import win32file
import struct
from grabscreen import grab_screen
import numpy as np
import cv2
from threading import Thread
from threading import Lock
capture_region = (0,40,1280,720)
reshape_size = (int(720/2),int(1280/2))
fileHandle = win32file.CreateFile(r'\\.\pipe\Demo', win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, win32file.OPEN_EXISTING, 0, None)
def get_state():
    state = grab_screen(region = capture_region)
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state,(reshape_size[1],reshape_size[0]))
    #state = state.reshape(reshape_size[0],reshape_size[1],1)
    return state
states = []
while(1):
    left, data = win32file.ReadFile(fileHandle, 4)
    
    v = struct.unpack('f',data)
    #states.append(v)
    print(v)
    #states.append(v)
   