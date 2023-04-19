from math import sin, cos, radians, floor, ceil

class Car:
    def __init__(self, color, x_fl, y_fl, x_fr, y_fr, x_br, y_br, x_bl, y_bl, vel=2, deg=0):
    	#initialize the position of the car based on given coordinates
        self.x_fl, self.y_fl, self.x_fr, self.y_fr, self.x_br, self.y_br, self.x_bl, self.y_bl = x_fl, y_fl, x_fr, y_fr, x_br, y_br, x_bl, y_bl
        self.front = (self.x_fl + self.x_fr)/2, (self.y_fl + self.y_fr)/2
        self.back = (self.x_bl + self.x_br)/2, (self.y_bl + self.y_br)/2
        self.vertices = [(self.x_fl, self.y_fl), (self.x_fr, self.y_fr), (self.x_br, self.y_br), (self.x_bl, self.y_bl)]

        #initialize the orientation, velocity, and color of the car
        self.deg = deg
        self.vel = vel
        self.color = color


    def rotate(self, clockwise=True, forward=True, rot_val=4):
    	#rotate the car relative to its back-center point
        if rot_val == None:
            rot_val = self.vel

        if clockwise:
            angle = radians(rot_val)
        else:
            angle = -radians(rot_val)
        
        #update the coordinates of the car after the rotation
        self.x_fl, self.y_fl = cos(angle)*(self.x_fl - self.back[0]) - sin(angle)*(self.y_fl - self.back[1]) + self.back[0], sin(angle)*(self.x_fl - self.back[0]) + cos(angle)*(self.y_fl - self.back[1]) + self.back[1]
        self.x_fr, self.y_fr = cos(angle)*(self.x_fr - self.back[0]) - sin(angle)*(self.y_fr - self.back[1]) + self.back[0], sin(angle)*(self.x_fr - self.back[0]) + cos(angle)*(self.y_fr - self.back[1]) + self.back[1]
        self.x_br, self.y_br = cos(angle)*(self.x_br - self.back[0]) - sin(angle)*(self.y_br - self.back[1]) + self.back[0], sin(angle)*(self.x_br - self.back[0]) + cos(angle)*(self.y_br - self.back[1]) + self.back[1]
        self.x_bl, self.y_bl = cos(angle)*(self.x_bl - self.back[0]) - sin(angle)*(self.y_bl - self.back[1]) + self.back[0], sin(angle)*(self.x_bl - self.back[0]) + cos(angle)*(self.y_bl - self.back[1]) + self.back[1]
        
        self.front = (self.x_fl + self.x_fr)/2, (self.y_fl + self.y_fr)/2
        self.back = (self.x_bl + self.x_br)/2, (self.y_bl + self.y_br)/2
        
        if clockwise:
            self.deg -= rot_val
        else:
            self.deg += rot_val

        #additionally move the car forward linearly
        self.move_lin(forward)

    
    def move_lin(self, forward=True):
    	#determine if linear motion is forward or backward
        if forward:
            vel = self.vel
        else:
            vel = -self.vel

        #linearly move the car by its velocity and update its coordinates
        self.x_fl, self.y_fl = self.x_fl + cos(radians(self.deg))*vel, self.y_fl - sin(radians(self.deg))*vel
        self.x_fr, self.y_fr = self.x_fr + cos(radians(self.deg))*vel, self.y_fr - sin(radians(self.deg))*vel
        self.x_bl, self.y_bl = self.x_bl + cos(radians(self.deg))*vel, self.y_bl - sin(radians(self.deg))*vel
        self.x_br, self.y_br = self.x_br + cos(radians(self.deg))*vel, self.y_br - sin(radians(self.deg))*vel
        
        self.front = (self.x_fl + self.x_fr)/2, (self.y_fl + self.y_fr)/2
        self.back = (self.x_bl + self.x_br)/2, (self.y_bl + self.y_br)/2
        
        #obtain the vertices of the car
        self.vertices = [(self.x_fl, self.y_fl), (self.x_fr, self.y_fr), (self.x_br, self.y_br), (self.x_bl, self.y_bl)]
