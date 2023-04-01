import pygame
import random

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
    self.x = SCREEN_WIDTH / 2
    self.y = SCREEN_HEIGHT / 2
    self.direction = "right"
    self.body = [(self.x, self.y), (self.x - BLOCK_SIZE, self.y),
                 (self.x - 2 * BLOCK_SIZE, self.y)]

  def move(self):
    if self.direction == "right":
      self.x += BLOCK_SIZE
    elif self.direction == "left":
      self.x -= BLOCK_SIZE
    elif self.direction == "up":
      self.y -= BLOCK_SIZE
    elif self.direction == "down":
      self.y += BLOCK_SIZE

    self.body.insert(0, (self.x, self.y))
    self.body.pop()

  def draw(self):
    for x, y in self.body:
      pygame.draw.rect(screen, VERDE, (x, y, BLOCK_SIZE, BLOCK_SIZE))

  def grow(self):
    self.body.append(self.body[-1])


class Food:

  def __init__(self):
    self.x = random.randint(0, SCREEN_WIDTH / BLOCK_SIZE - 1) * BLOCK_SIZE
    self.y = random.randint(0, SCREEN_HEIGHT / BLOCK_SIZE - 1) * BLOCK_SIZE

  def draw(self):
    pygame.draw.rect(screen, ROJO, (self.x, self.y, BLOCK_SIZE, BLOCK_SIZE))


# Definir funciones
def game_over():
  font = pygame.font.SysFont(None, 50)
  text = font.render("Game Over", True, WHITE)
  screen.blit(text, (SCREEN_WIDTH / 2 - 75, SCREEN_HEIGHT / 2 - 25))
  pygame.display.update()
  pygame.time.delay(2000)
  pygame.quit()
  quit()


# Inicializar objetos
snake = Snake()
food = Food()

# Bucle principal del juego
clock = pygame.time.Clock()
while True:
  clock.tick(FPS)

  # Manejo de eventos
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      quit()
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_RIGHT and snake.direction != "left":
        snake.direction = "right"
      elif event.key == pygame.K_LEFT and snake.direction != "right":
        snake.direction = "left"
      elif event.key == pygame.K_UP and snake.direction != "down":
        snake.direction = "up"
      elif event.key == pygame.K_DOWN and snake.direction != "up":
        snake.direction = "down"

  # Actualizar la serpiente y la comida
  snake.move()
  if snake.body[0][0] == food.x and snake.body[0][1] == food.y:
    snake.grow()
    food = Food()
  if snake.body[0][0] < 0 or snake.body[0][0] >= SCREEN_WIDTH or snake.body[0][
      1] < 0 or snake.body[0][1] >= SCREEN_HEIGHT or snake.body[
        0] in snake.body[1:]:
    game_over()

  # Dibujar la pantalla
  screen.fill(GRIS)
  snake.draw()
  food.draw()
  pygame.display.update()
