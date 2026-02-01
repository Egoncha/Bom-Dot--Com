#speaker file

# import os
#
# def play(musicFile):
#     os.system(f"aplay {musicFile}")

#py game implementation

import pygame 
import time 


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
    pygame.mixer.music.load("vine-boom.mp3")
    pygame.mixer.music.play()

    time.sleep(2)
    
    pygame.mixer.music.stop()
    pygame.mixer.quit()
