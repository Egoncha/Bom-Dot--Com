#speaker file

# import os
#
# def play(musicFile):
#     os.system(f"aplay {musicFile}")

#py game implementation

import pygame 
import time 
import random


# pygame.mixer.init()
# pygame.mixer.music.load("vine-boom.mp3")
# pygame.mixer.music.play()
#
# time.sleep(2)          # play duration in seconds
#
# pygame.mixer.music.stop()
# pygame.mixer.quit()

def play_sound():
    pygame.mixer.init()
    choice = random.randint(1, 3)
    if choice == 1:
        pygame.mixer.music.load("Sampada.mp3")
    elif choice == 2:
        pygame.mixer.music.load("Intia.mp3")
    elif choice == 3:
        pygame.mixer.music.load("Ethan.mp3")
    pygame.mixer.music.play()
    time.sleep(2)
    pygame.mixer.music.stop()
    pygame.mixer.quit()
