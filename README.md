# DinosaurGameAI
go to the OCR branch if you want a training to be less buggy and its just faster overall

This repository uses opencv and NEAT to detect obstacles in the dinosaur game, feeds the information to a NEAT trained neural network to play a clone of Google's dinosaur offline game (http://www.trex-game.skipser.com/) almost flawlessly. 

If you want to use the already trained model, run dinogame_winner.py, or you could train you own model using dinogame_NEAT.py or dinogameVision_NEAT.py. If you want to see how the program works, you can use the dinogameVision_NEAT.py to train, but when training for long periods of time (3hr+) use dinogame_NEAT.py.

For the program to recognize the game, you should place chrome in the top left corner of the screen. Use dinogame.py to make sure that all obstacles are recognized with red. Note: Make sure when the dinosaur is crounching, the program does not recognize it as an obstacle. 

In later versions, I will try to make this easier to use.



UPDATE 1: #1 on leaderboard as of 4/5/2020 with a score of 43035 (refreshes every week)
Unfortunately did not get to save that neural network, but saved a 23k nn, which you can run with dinogame_winner.py. When I train a better nn I will upload it

Dependencies(there's a lot): graphviz, matplotlib, numpy, opencv, mss, pyautogui, tesseract, python-neat

