# https://github.com/Box-Of-Hats

import win32api as wapi

keyList=["\b"]
for char in "ABCDEFGHIJKLMNOPRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)
    
def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
