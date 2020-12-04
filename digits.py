from my_CNN_model import *
import time
import cv2
import numpy as np
import pygame
from PIL import Image
from keras import backend as K


pygame.init()


def initialize_screen(scr):
    screen.fill(background_colour)
    pygame.draw.rect(scr, (255, 255, 255), (0, WINDOW_SIZE, WINDOW_SIZE, HEIGHT - WINDOW_SIZE))


background_colour = (0, 0, 0)
my_model = load_my_CNN_model('my_model')
WINDOW_SIZE = 448
# screen.fill(background_colour)
# pygame.display.flip()
time.sleep(2.4)
HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, HEIGHT))
pygame.display.set_caption("Digit Predictor")
initialize_screen(screen)
img_rows, img_cols = 28, 28

clock = pygame.time.Clock()
size = 28, 28
loop = True
press = False
blue=0
green=0
def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)

def show_result(screen, result):
    font = pygame.font.SysFont('Comic Sans MS', 30)
    text = font.render('Number Guessed is {}'.format(result), True, (0, 0, 0), (255, 255, 255))
    textRect = text.get_rect()
    textRect.center = (WINDOW_SIZE / 2, WINDOW_SIZE + 75)
    screen.blit(text, textRect)


while loop:
    try:
        # pygame.mouse.set_visible(False)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loop = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    # pygame.image.save(screen, "digit.png")

                    rect = pygame.Rect(0, 0, WINDOW_SIZE, WINDOW_SIZE)
                    sub = screen.subsurface(rect)
                    pygame.image.save(sub, "digit.png")

                    im = Image.open('digit.png')
                    im.thumbnail(size, Image.ANTIALIAS)
                    im.save('digit.png', "PNG")
                    # im = Image.open('digit.png')

                    gray = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)
                    gray = cv2.resize(gray, (28, 28))

                    new_gray = np.array([[[0]]*28]*28)
                    np.reshape(new_gray, (28, 28, 1))

                    for row in range(len(gray)):
                        for pixel in range(len(gray[row])):
                            num = gray[row][pixel]
                            new_gray[row][pixel] = [num]

                    to_predict = new_gray
                    np.reshape(to_predict, (28, 28, 1))
                    # if K.image_data_format() == 'channels_first':
                    #     to_predict = to_predict.reshape(to_predict, 1, img_rows, img_cols)
                    #     input_shape = (1, img_rows, img_cols)
                    # else:
                    #     to_predict = to_predict.reshape(to_predict.shape[0], img_rows, img_cols, 1)
                    #     input_shape = (img_rows, img_cols, 1)

                    to_predict = to_predict.astype('float32')
                    to_predict /= 255
                    # print(to_predict)
                    to_predict = np.array([to_predict])
                    # digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
                    # digit_resized_copy = digit_resized.copy()
                    # digit_resized = digit_resized.reshape(1, 96, 96, 1)

                    prediction = my_model.predict(to_predict)
                    # print(prediction)

                    result = prediction.tolist()[0].index(max(prediction.tolist()[0]))
                    initialize_screen(screen)
                    show_result(screen, result)

        px, py = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed() == (1, 0, 0):
            # pygame.draw.circle(screen, (255, 255, 255), (px, py, 10, 10))
            pygame.draw.circle(screen, (255, 255, 255), (px, py), 12)
        pygame.display.update()
        clock.tick(1000)
    except StopIteration as e:
        print("Error", e)
        pygame.quit()

pygame.quit()






# Cleanup the camera and close any open windows
# cv2.destroyAllWindows()
