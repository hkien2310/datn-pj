from djitellopy import tello
import pygame
import cv2

datn = tello.Tello()
datn.connect()
print(datn.get_battery())

def init():
    pygame.init()
    wins = pygame.display.set_mode((500,500))

def getKey(Keyname):
    ans = False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(Keyname))
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if getKey('LEFT'): lr = -speed
    elif getKey('RIGHT'): lr = speed

    if getKey('UP'): fb = speed
    elif getKey('DOWN'): fb = -speed

    if getKey('w'): ud = speed
    elif getKey('s'): ud = -speed

    if getKey('a'): yv = -speed
    elif getKey('d'): yv = speed

    if getKey('q'): datn.land()

    if getKey('e'): datn.takeoff()

    if getKey('o'): datn.streamon()

    if getKey('p'): datn.streamoff()

    return [lr, fb, ud, yv]


init()

while True:
    gtri = getKeyboardInput()
    datn.send_rc_control(gtri[0], gtri[1], gtri[2], gtri[3])
    if getKey('o'):
        while True:
            gtri = getKeyboardInput()
            datn.send_rc_control(gtri[0], gtri[1], gtri[2], gtri[3])
            cam = datn.get_frame_read().frame
            cam = cv2.resize(cam,(480,360))
            cv2.imshow("Camera", cam)
            cv2.waitKey(1)
            if getKey('p'): break
    if getKey('b'): break
