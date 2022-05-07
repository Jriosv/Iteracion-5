from tensorflow import keras
from q_network import Q_model
from chess_env import *
from board_conversion import *
import numpy as np

env = ChessEnv()
q_model = Q_model(keras.models.load_model("./checkpoint3"))

p, v = q_model.predict([env.board])

best_move = np.argmax(p, axis=None)
move = num2move[best_move]
print("BEST_MOVE:", move) 