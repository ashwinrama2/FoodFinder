import pygame
import numpy as np
from math import sin, cos, radians
from random import randint
from car import Car
from calc import poly_points, distance

pygame.init()

colors = {   
            "black"  : (0, 0, 0),
            "grey"   : (70, 70, 70), 
            "blue"   : (0, 0, 255),
            "green"  : (0, 255, 0), 
            "yellow" : (253, 218, 22), 
            "red"    : (255, 0, 0), 
            "beige"  : (247, 233, 210),
            "white"  : (255, 255, 255),
        }


class CarGameAI:
    def __init__(self, w=600, h=400):
    	#initialize the Pygame GUI frame
        self.w = w
        self.h = h
        self.gameDisplay = pygame.display.set_mode((self.w, self.h))
        self.clock=pygame.time.Clock()
        self.reset()


    def reset(self):
    	#initialize the car object on the frame
        self.car = Car("red", 140, 180, 140, 200, 100, 200, 100, 180, vel=2, deg=0)
        self.place_food()


    def place_food(self):
        x = randint(10, self.w - 10) #left and right x-coordinates of horiz road (tolarance of 10)
        y = randint(150, 250) #top and bottom y-coordinates of road
        self.min_dist = float("inf") #minimum distance so far to food
        self.frame_iteration = 0
        self.food = (x, y) #location of center of food pellet
        if self.ate_food(): #place food anywhere in road not colliding with car
            self.place_food()


    def is_collision(self): #check if vertexes of car has collision with boundary of frame or goes off-road
        for x, y in self.car.vertices:
            if x >= self.w or x <= 0 or y >= 260 or y <= 140:
                return True
        return False


    def view_ahead_pt(self, dist, angle): #check if a point ahead of car front-center points has off-road collision or frame boundary collision
        x, y = self.car.front
        x_new = x + cos(radians(self.car.deg + angle)) * dist
        y_new = y - sin(radians(self.car.deg + angle)) * dist

        if x_new >= self.w or x_new <= 0 or y_new >= 260 or y_new <= 140: #boundaries of road
            return 0
        elif distance((x_new, y_new), self.food) < 5: #x_new, y_new near food
            return 2
        else:
            return 1


    def ate_food(self): #checks if FRONT (left, right, or center) of the car has intersected with the food
        car_front_right = int(self.car.x_fr), int(self.car.y_fr)
        car_front_left = int(self.car.x_fl), int(self.car.y_fl)
        car_front_center = int(self.car.front[0]), int(self.car.front[1])
        return distance(car_front_center, self.food) < 5 or distance(car_front_right, self.food) < 5 or distance(car_front_left, self.food) < 5


    def draw_point(self, center, size=1, color="green"): #draw a point at position (x, y)
        x, y = center
        pygame.draw.polygon(self.gameDisplay, colors[color], [(x-size, y-size), (x+size, y-size), (x+size, y+size), (x-size, y+size)])


    def draw_road(self): #draw the road using Pygame polygons customized to exact pixel locations
        y = 150
        pygame.draw.polygon(self.gameDisplay, colors["grey"], [(0, y-10), (self.w, y-10), (self.w, y+110), (0, y+110)])
        pygame.draw.polygon(self.gameDisplay, colors["white"], [(0, y), (self.w, y), (self.w, y+1), (0, y+1)])
        for i in range(0, self.w, 40):
            pygame.draw.polygon(self.gameDisplay, colors["white"], [(i, y+31), (i+10, y+31), (i+10, y+32), (i, y+32)])
            pygame.draw.polygon(self.gameDisplay, colors["white"], [(i+5, y+64), (i+15, y+64), (i+15, y+65), (i+5, y+65)])
        pygame.draw.polygon(self.gameDisplay, colors["white"], [(0, y+100), (self.w, y+100), (self.w, y+101), (0, y+101)])


    def update_ui(self, perception=True): #generate the GUI on Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.gameDisplay.fill(colors["beige"]) #draw background
        self.draw_road() #draw road/infrastructure
        pygame.draw.polygon(self.gameDisplay, colors[self.car.color], self.car.vertices) #draw car
        pygame.draw.rect(self.gameDisplay, colors["yellow"], pygame.Rect(self.food[0]-5, self.food[1]-5, 10, 10)) #draw food pellet
        
        #if user enables perception, draw perception samples
        if perception:
            x, y = self.car.front
            for angle in range(-90, 91, 15):
                for freq in range(20, 121, 20):
                    x_new = x + cos(radians(self.car.deg+angle))*freq
                    y_new = y - sin(radians(self.car.deg+angle))*freq
                    if distance((x_new, y_new), self.food) < 5:
                        self.draw_point((x_new, y_new), 1, "red")
                    else:
                        self.draw_point((x_new, y_new), 1)

        pygame.display.flip()


    def play_step(self, action):
    	#given an action passed by the agent, perform that action
        self.frame_iteration += 1
        self.move(action)
        
        reward = 0
        game_over = False

        #end the game upon collision
        if self.is_collision() or self.frame_iteration > 1500:
            game_over = True
            reward = -15
            return reward, game_over

        #update the reward according to food and distance heuristics
        if self.ate_food():
            reward = 15
            self.place_food()
        else: 
            dst = distance(self.car.front, self.food)
            if dst < self.min_dist: #reward when car gets closer to food
                self.min_dist = dst 
                reward = 2
            else:
                reward = -5 #penalize when car gets further from food

        self.clock.tick(1000)
        self.update_ui()
        return reward, game_over


    def move(self, action):
        if   np.array_equal(action, [1, 0, 0, 0, 0, 0]):
            self.car.move_lin(True) #linear forward
        elif np.array_equal(action, [0, 1, 0, 0, 0, 0]):
            self.car.move_lin(False) #linear backward
        elif np.array_equal(action, [0, 0, 1, 0, 0, 0]):
            self.car.rotate(True, True, 6) #rotate clockwise and forward
        elif np.array_equal(action, [0, 0, 0, 1, 0, 0]):
            self.car.rotate(False, True, 6) #rotate counter-clockwise and forward
        elif np.array_equal(action, [0, 0, 0, 0, 1, 0]):
            self.car.rotate(True, False, 6) #rotate clockwise and backward
        elif np.array_equal(action, [0, 0, 0, 0, 0, 1]):
            self.car.rotate(False, False, 6) #rotate counter-clockwise and backward
        else:
            pass