""" Does not include visualization of the opencv algorithm """
# for visualization, run dinogameVision_NEAT.py

import numpy as np
import cv2
import time
import mss # for getting screenshots efficiently
import find
import pyautogui
import pytesseract
import os
import neat
import visualize # provided by neat
import pickle
import template_matching as ocr

print('starting')

# title of our window
title = "Computer Vision"
# set start time to current time
start_time = time.time()
# displays the frame rate every .2 second
display_time = 0.2
fps = 0
sct = mss.mss()
# Set monitor size to capture to MSS
high = 9999999999
monitor = {"top": 265, "left": 210, "width": 590, "height": 140}
update_dino = 1 # how many frames until you update the dino's postion
frame = 0
control_frame_interval = 1
max_fitness = -1

cac_mask_upper = np.array([0, 0, 90])
cac_maks_lower = np.array([0, 0, 10])

contours = 0

ROI = [(43, 256), (136, 1300)]

time_pixel_buffer = 5
img = np.array(sct.grab(monitor))
time.sleep(2)
play_toggle = True
games = 10
game = 0
scores = []
score_ROI = [(0, 80), (1005, 1200)]

generation = 0

def findBoundingBoxesROI(contours):
    return_array = []
    for obstacle in contours:
        max_x = -1
        max_y = -1
        min_x = high
        min_y = high

        for coord in obstacle:
            coord = coord[0]

            max_x = coord[0] if coord[0] > max_x else max_x
            max_y = coord[1] if coord[1] > max_y else max_y
            min_x = coord[0] if coord[0] < min_x else min_x
            min_y = coord[1] if coord[1] < min_y else min_y

        return_array.append([(int(min_x), int(max_y)), (int(max_x), int(min_y))])

    return return_array

def findBoundingBoxesWithShift(contours):
    global ROI
    return_array = []
    for obstacle in contours:
        max_x = -1
        max_y = -1
        min_x = high
        min_y = high

        for coord in obstacle:
            coord = coord[0]

            max_x = coord[0] if coord[0] > max_x else max_x
            max_y = coord[1] if coord[1] > max_y else max_y
            min_x = coord[0] if coord[0] < min_x else min_x
            min_y = coord[1] if coord[1] < min_y else min_y

        return_array.append([(min_x + ROI[1][0], max_y + ROI[0][0]), (max_x + ROI[1][0], min_y + ROI[0][0])])

    return return_array

def drawBoundingBoxes(img, points, color):

    for coord in points:
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, coord[0], coord[1], color, 2)

    return img

def findDistance(dino_coords, obstacles):
    dino_x = dino_coords[0][1][0]
    dino_mid_y = (dino_coords[0][0][1] + dino_coords[0][1][1]) / 2 # 226 when running
    clump = []

    obstacles = sorted(obstacles, key=lambda obstacle: obstacle[0][0])
    for i, obstacle in enumerate(obstacles):
        if obstacle[1][1] < 145:
            obstacles.pop(i)

    obstacles = sorted(obstacles, key=lambda obstacle: obstacle[0][0])

    cac_x = 1300
    if len(obstacles) > 0:
        closest = obstacles[0]
        cac_x = obstacles[0][0][0]
    else:
        closest = [[0, dino_mid_y], [0, dino_mid_y]]

    index = 0

    clump.append(closest)
    while index + 1 < len(obstacles):
        c1_x = obstacles[index][1][0]
        c2_x = obstacles[index+1][0][0]

        if c2_x - c1_x < 15:
            clump.append(obstacles[index+1])
        else:
            break
        index += 1
    width = 0

    if len(clump) == 1:
        width = clump[0][1][0] - clump[0][0][0]
    else:
        width = clump[len(clump)-1][1][0] - clump[0][0][0]

    cac_mid_y = (closest[0][1] + closest[1][1]) / 2
    cac_x -= time_pixel_buffer
    dist = cac_x - dino_x - time_pixel_buffer
    return dist, (int(dino_x), int(dino_mid_y)), (int(cac_x), int(cac_mid_y)), width

def reload():
    pyautogui.keyDown('command')
    pyautogui.press('r')
    pyautogui.keyUp('command')

def calibrate():
    global frame, ROI, play_toggle, game, score_ROI
    last_dist = 0
    tess_config = ('-l eng --oem 1 --psm 3')
    gameover_ROI = [(50, 120), (300, 900)]
    score_time = time.time()
    force_gameover = False
    print('Focus the chrome browser to start calibration')
    time.sleep(5)
    print('calibrating')

    print('pressing up')
    pyautogui.press('up')
    print('pressed up')
    time.sleep(1)
    pyautogui.press('up')
    while True:
        img = np.array(sct.grab(monitor))
        obstacles = img[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]
        # obstacles = img[43:256, 950:1300]
        gray = cv2.cvtColor(obstacles, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,90,255,cv2.THRESH_BINARY)

        edged = cv2.Canny(thresh, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
        points = findBoundingBoxesWithShift(contours)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dino_coords = find.find_dino(img)

        # print(dist)
        # print(start_point, end_point)

        # if dist > 20 and dist < 260:


        if (len(points) >= 9 and len(points) < 20) or force_gameover:
            # Run tesseract OCR on image
            gameover = img[gameover_ROI[0][0]:gameover_ROI[0][1], gameover_ROI[1][0]:gameover_ROI[1][1]]
            text = pytesseract.image_to_string(gameover, config=tess_config)
            # print('maybe gameover')
            if (text.lower().replace(' ', '') == "gameover") or force_gameover:


                score_img = img[score_ROI[0][0]:score_ROI[0][1], score_ROI[1][0]:score_ROI[1][1]]
                # gray_score = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(score_img,90,255,cv2.THRESH_BINARY)
                score = ocr.get_score(thresh)
                time_score = time.time() - score_time
                try:
                    print(f'Game Over! Game: {game} - Score: {int(score)}, Time Score: {time_score}', end='\n\n')
                except:
                    print(f"{score} is not an integer")

                if time_score >= 3:
                    game += 1

                    break

        if score_time - time.time() > 400:
            force_gameover = True

def eval_genomes(genomes, config):
    global frame, ROI, play_toggle, game, score_ROI, generation, max_fitness, winner
    print('playing')
    last_dist = 0
    tess_config = ('-l eng --oem 1 --psm 7')
    score_tess_config = ('digits --oem 2 --psm 5')
    gameover_ROI = [(50, 120), (300, 900)]
    score_time = time.time()
    force_gameover = False
    generation += 1
    fgo_thresh = 350 + generation * 50
    fgo_thresh = 1800 if fgo_thresh > 1800 else fgo_thresh

    reload()
    print('reloaded page')

    time.sleep(1)

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        scroll_go = False
        force_gameover = False
        score_time = time.time()

        # start game
        pyautogui.press('up')
        time.sleep(1)
        pyautogui.press('up')
        while True:
            # Get raw pixels from the screen, save it to a Numpy array
            try:
                # screen shot
                img = np.array(sct.grab(monitor))

                # roi of just obstacles
                obstacles = img[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]
                # obstacles = img[43:256, 950:1300]
                gray = cv2.cvtColor(obstacles, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray,90,255,cv2.THRESH_BINARY) # get binary image of abstacles

                edged = cv2.Canny(thresh, 30, 200) # edge detection to find obstacle hitboxes
                # find contours of obstacles
                contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # get hitboxes
                points = findBoundingBoxesWithShift(contours)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                dino_coords = find.find_dino(img)

                # find dist between dino and nearest obstacle
                dist, start_point, end_point, width = findDistance(dino_coords, points)

                # feed data to nn and run output
                dino_y = start_point[-1]
                cac_y = end_point[-1]
                delta_dist = abs(last_dist - dist)
                output = net.activate([dist, cac_y, dino_y, width, delta_dist])

                # [0] jump, [1] crouch, [2] idle
                if output[0] >= output[1] and output[0] >= output[2]: # if jump is the highest value
                    pyautogui.press('up')
                elif output[1] >= output[0] and output[1] >= output[2]:
                    pyautogui.press('down')
                    genome.fitness += 0.5
                    if cac_y > 310:
                        genome.fitness += 150
                elif output[2] >= output[0] and output[2] >= output[1]:
                    pass

                # check if game ended
                if (len(points) >= 9 and len(points) < 20 and last_dist == dist) or force_gameover:
                    # Run tesseract OCR on image to see if words "game over" are on screen
                    gameover = img[gameover_ROI[0][0]:gameover_ROI[0][1], gameover_ROI[1][0]:gameover_ROI[1][1]]
                    text = pytesseract.image_to_string(gameover, config=tess_config)
                    # print('maybe gameover')
                    if (text.lower().replace(' ', '') == "gameover") or force_gameover:

                        # for when nn presses down and scrolls under game over text
                        if scroll_go:
                            pyautogui.scroll(20, x=690, y=450)
                            time.sleep(1)
                            pyautogui.scroll(20, x=690, y=450)
                            print('scrolling')
                            time.sleep(0.3)

                        # total time the dino has ran for
                        time_score = time.time() - score_time

                        # make sure score isnt ''
                        score = ''
                        count = 0
                        while score == '' and count < 5:
                            pyautogui.scroll(10, x=690, y=450)
                            img = np.array(sct.grab(monitor))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            score_img = img[score_ROI[0][0]:score_ROI[0][1], score_ROI[1][0]:score_ROI[1][1]]
                            # gray_score = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
                            ret, thresh = cv2.threshold(score_img,90,255,cv2.THRESH_BINARY)
                            score = ocr.get_score(thresh)
                            time.sleep(0.7)
                            count += 1

                        if count >= 4:
                            score = 0

                        print(f'Bonus: {genome.fitness}')
                        try:
                            print(f'Game Over! Game: {game} - \nScore: {int(score)}\n Time: {time_score}\n Fitness: {genome.fitness + int(score)}', end='\n\n')
                        except:
                            print(f"{score} is not an integer")

                        # print(f'Upper: {time_score * ((int(score)/100)+20)}, Lower: {time_score * 5}', end='\n\n')
                        if time_score < 1800 and False:
                            if int(score) > time_score * ((int(score)/100)+13):
                                score = (time_score - 2) * 10
                                print(f'Adjusted Score: {score}')
                            elif int(score) < time_score * 5:
                                score = (time_score) * 8
                                print(f'Adjusted Score: {score}')

                        # make sure game lasted over three seconds
                        if time_score >= 3:
                            game += 1
                            if not force_gameover or scroll_go:
                                # set fitness
                                genome.fitness += int(score)
                                if genome.fitness > max_fitness:
                                    winner = genome
                                break
                            else:
                                # replay game
                                reload()
                                pyautogui.scroll(20, x=690, y=450)
                                pyautogui.press('up')
                                time.sleep(1)
                                pyautogui.press('up')
                                score_time = time.time()

                                force_gameover = False

                if len(points) > 25: # if nn spams down and scrolls the page
                    # pyautogui.scroll(20, x=690, y=450)
                    force_gameover = True
                    scroll_go = True
                    print('scroll gameover')
                    time.sleep(0.5)

                # when game cant see the game over text, it times out
                if score_time - time.time() > fgo_thresh:
                    print('timeout')
                    force_gameover = True

                if int(time.time())%1000 == 0:
                    pyautogui.scroll(-10, x=690, y=450)
                    pyautogui.scroll(30, x=690, y=450)
                    print('auto scrolling')
                    if time.time() - score_time > 1800:
                        reload()
                        pyautogui.press('up')
                        time.sleep(1)
                        pyautogui.press('up')
                        score_time = time.time()


                last_dist = dist

            except KeyboardInterrupt:
                print('program terminated: keyboard interrupt')
                cv2.destroyAllWindows()
                break

winner = ''
stats = ''
config = None
def run(config_path):
    global winner, stats, config
    calibrate() # make sure game starts the same way everytime
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-6')
    # print('restored population')

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    try:
        winner = p.run(eval_genomes, 100) # run for up to 100 generations
    except Exception as e:
        print(e)

    # print('DONE TRAINING')
    with open('winner-ctrnn-new', 'wb') as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, ylog=True, view=True, filename="ctrnn-fitness.svg")

    try:
        visualize.plot_species(stats, view=True, filename="ctrnn-speciation.svg")
    except ValueError:
        print('An entire generation has not passed yet')

    if winner != '':
        node_names = {-1: 'distance', -2: 'obstacle y', -3: 'dino y', -4: 'width', -5: 'speed', -6: 'dtime', 0: 'jump', 1: 'crouch', 2: 'idle'}
        visualize.draw_net(config, winner, True, node_names=node_names)

        visualize.draw_net(config, winner, view=True, node_names=node_names,
                           filename="winner-ctrnn-new.gv")
        visualize.draw_net(config, winner, view=True, node_names=node_names,
                           filename="winner-ctrnn-new-enabled.gv", show_disabled=False)
        visualize.draw_net(config, winner, view=True, node_names=node_names,
                           filename="winner-ctrnn-new-enabled-pruned.gv", show_disabled=False, prune_unused=True)

    # print('\nBest Dino: \n{!s}'.format(winner))



if __name__ == '__main__':
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    # show_vision()
    run(config_path)
