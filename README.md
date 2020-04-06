# DinosaurGameAI
This repository uses opencv and NEAT to detect obstacles in the dinosaur game, feeds the information to a NEAT trained neural network to play a clone of Google's dinosaur offline game (http://www.trex-game.skipser.com/) almost flawlessly. 

If you want to use the already trained model, run [will be avaliable in the future].py, or you could train you own model using dinogame_NEAT.py.

For the program to recognize the game, you should place chrome in the top left corner of the screen. Use dinogame.py to make sure that all obstacles are recognized with red. Note: Make sure when the dinosaur is crounching, the program does not recognize it as an obstacle. 

In later versions, I will try to make this easier to use.

Dependencies(there's a lot): graphviz, matplotlib, numpy, opencv, mss, pyautogui, tesseract, python-neat

