import pygame
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam


# Inicializar Pygame
pygame.init()

# Definir constantes
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
BLOCK_SIZE = 10
FPS = 10

# Crear la ventana
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")

# Definir colores
MORADO = (74, 30, 127)
VERDEOSCURO = (50, 61, 25)
VERDE = (121, 162, 82)
ROJO = (99, 32, 36)
GRIS = (48, 47, 48)
WHITE = (205, 188, 169)


# Definir clases
class Snake:

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = SCREEN_WIDTH / 2
        self.y = SCREEN_HEIGHT / 2
        self.direction = "right"
        self.body = [(self.x, self.y), (self.x - BLOCK_SIZE, self.y),
                     (self.x - 2 * BLOCK_SIZE, self.y)]

    def move(self):
        if self.direction == "right":
            new_head = (self.body[0][0] + BLOCK_SIZE, self.body[0][1])
        elif self.direction == "left":
            new_head = (self.body[0][0] - BLOCK_SIZE, self.body[0][1])
        elif self.direction == "up":
            new_head = (self.body[0][0], self.body[0][1] - BLOCK_SIZE)
        elif self.direction == "down":
            new_head = (self.body[0][0], self.body[0][1] + BLOCK_SIZE)

        self.body.insert(0, new_head)
        self.body.pop()

    def grow(self):
        self.body.append(self.body[-1])



    def get_state(self, food): 
        state = [self.body[0][0], self.body[0][1], food.x, food.y]
        return np.array(state)

    def step(self, action, food):  
        if action == 0 and self.direction != "left":
            self.direction = "right"
        elif action == 1 and self.direction != "right":
            self.direction = "left"
        elif action == 2 and self.direction != "down":
            self.direction = "up"
        elif action == 3 and self.direction != "up":
            self.direction = "down"

        self.move()

        reward = 0
        done = False

        if self.body[0][0] == food.x and self.body[0][1] == food.y: 
            self.grow()
            reward = 1
            food.x, food.y = food.new_pos()  
            food = Food()  
        if (self.body[0][0] < 0 or self.body[0][0] >= SCREEN_WIDTH or self.body[0][1] < 0 or
            self.body[0][1] >= SCREEN_HEIGHT or self.body[0] in self.body[1:]):
            reward = -1  
            done = True

        return self.get_state(food), reward, done  




class Food:

  def __init__(self):
    self.x = random.randint(0, SCREEN_WIDTH / BLOCK_SIZE - 1) * BLOCK_SIZE
    self.y = random.randint(0, SCREEN_HEIGHT / BLOCK_SIZE - 1) * BLOCK_SIZE

  def draw(self):
    pygame.draw.rect(screen, ROJO, (self.x, self.y, BLOCK_SIZE, BLOCK_SIZE))
  def new_pos(self):
    new_x = random.randint(0, (SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
    new_y = random.randint(0, (SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
    return new_x, new_y




# Definir funciones
def draw_grid():
  for x in range(0, SCREEN_WIDTH, BLOCK_SIZE):
    pygame.draw.line(screen, GRIS, (x, 0), (x, SCREEN_HEIGHT))
  for y in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
    pygame.draw.line(screen, GRIS, (0, y), (SCREEN_WIDTH, y))

def draw_snake(snake):
  for block in snake.body:
    pygame.draw.rect(screen, VERDE, (block[0], block[1], BLOCK_SIZE, BLOCK_SIZE))


def game_over():
  font = pygame.font.SysFont(None, 50)
  text = font.render("Game Over", True, WHITE)
  screen.blit(text, (SCREEN_WIDTH / 2 - 75, SCREEN_HEIGHT / 2 - 25))
  pygame.display.update()
  pygame.time.delay(2000)
  pygame.quit()
  quit()



# Inicializar el juego de Snake
class SnakeAI:

  def __init__(self):
    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(
      LSTM(64, activation='relu', return_sequences=True,
           input_shape=(None, 4)))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(4, activation='linear'))
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model

  def predict(self, state):
    return np.argmax(self.model.predict(state[np.newaxis, :, np.newaxis]))



  def train(self, x_train, y_train, epochs=100):
    x_train = x_train[:, np.newaxis, :]  # Add this line to reshape x_train
    self.model.fit(x_train, y_train, epochs=epochs)


  def save(self, filename):
    self.model.save(filename)

  def load(self, filename):
    self.model.load_weights(filename)


def generate_training_data(snake_game, food, num_games=1000, num_steps=1000):
    x_train = []
    y_train = []

    for _ in range(num_games):
        snake_game.reset()

        for _ in range(num_steps):
            state = snake_game.get_state(food)  # Pass the food instance here
            action = random.randint(0, 3)  # Select a random action
            next_state, reward, done = snake_game.step(action, food)

            x_train.append(state)
            y_train.append(action)

            if done:
                break

    return np.array(x_train), np.array(y_train)


def main():
  ai = SnakeAI()
      
  # Inicializar el juego de Snake
  snake = Snake()  # Cambiar de snake_game a snake
  food = Food()
  snake_ai = SnakeAI()

  # Generar datos de entrenamiento
  x_train, y_train = generate_training_data(snake, food)

  # Entrenar el modelo
  ai.train(x_train, y_train)

  # Guardar el modelo entrenado
  ai.save("snake_ai_model.h5")

  # Cargar el modelo entrenado
  ai.load("snake_ai_model.h5")

  clock = pygame.time.Clock()
  while True:
    clock.tick(FPS)

  # Manejo de eventos
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

  # Generar el estado actual del juego
    state = snake.get_state(food)

  # Predecir la mejor acción utilizando el modelo entrenado
    action = snake_ai.predict(state)

  # Realizar la acción predicha en el juego
    next_state, reward, done = snake.step(action, food)

  # Dibujar la pantalla
    screen.fill(GRIS)
    draw_grid()
    draw_snake(snake)  # Cambiar de snake.draw(snake) a draw_snake(snake)
    food.draw()  # Dibujar la comida
    pygame.display.flip()

    if done:
      pygame.time.delay(500)
      snake.reset()
      food = Food()

if __name__ == "__main__":
    main()
