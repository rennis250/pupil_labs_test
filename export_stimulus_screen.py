from psychopy import visual, event
import os
import pandas as pd
import numpy as np
import robsblobs

monGamma_sRGB = np.array([2.2, 2.2, 2.2])
monxyY_sRGB = np.array([
    [0.6400, 0.3300, 0.2126*100],
    [0.3000, 0.6000, 0.7152*100],
    [0.1500, 0.0600, 0.0722*100]])
mon_sRGB = robsblobs.monitor.Monitor("sRGB")
mon_sRGB.set_monGamma(monGamma_sRGB)
mon_sRGB.set_monxyY(monxyY_sRGB)

win = visual.Window(fullscr=True, color=(0, 0, 0), units="pix")
win_size = win.size
aspr_correction_x = win_size[1]/win_size[0]

instructions = visual.TextStim(win, text="Match the circle's color to the image and press space to continue.")
instructions.draw()
win.flip()
event.waitKeys(keyList=["space"])

win.mouseVisible = False

image_dir = 'images'
image_list = os.listdir(image_dir)

mouse = event.Mouse(win=win)

final_rgb = robsblobs.dkl.dkl2rgb(mon_sRGB, np.array([-0.15000000000000008, 0.24375, 0.6611111111111111]))
circle = visual.Circle(win, radius=100, fillColor=np.sqrt(final_rgb), lineColor='black', lineWidth=2.5, pos=(win_size[0]/4, 0), colorSpace='rgb1')

# Set up a loop to present each image
for image_file in image_list:
    image_path = os.path.join(image_dir, image_file)
    image = visual.ImageStim(win, image=image_path, pos=(-400, 0), size=(600, 600))
    
    win.mouseVisible = False

    while True:
        image.draw()
        circle.draw()
        win.flip()

        win.getMovieFrame()

        break

win.saveMovieFrames('stimulus_screen.png')

win.close()