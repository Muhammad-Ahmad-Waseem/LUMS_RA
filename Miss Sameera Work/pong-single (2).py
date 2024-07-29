# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:55:16 2022

@author: Sameera Saleem

for use with eye-blink.py

"""



import pygame
from time import sleep



BLACK = (0,0,0)

WHITE = (255,255,255)

RED = (255,0,0)

GREEN = (0,255,0)

BLUE = (0,0,255)



pygame.init()

#author---Katha Roy



size = (760,600)

screen = pygame.display.set_mode(size)

pygame.display.set_caption("pong")





rect_x = 335

rect_y = 580





rect_change_x = 0

rect_change_y = 0





ball_x = 50

ball_y = 50




ball_change_x = 5

ball_change_y = 5



score = 0



#author---Katha Roy

def drawrect(screen,x,y):

    if x <= 0:

        x = 0

    if x >= 699:

        x = 699    

    pygame.draw.rect(screen,RED,[x,y,100,20])

     
   #author---Katha Roy

done = False
check = False
check1 = False
check2 = False
check3 = False
check4 = False
check5 = False
startGame = False

clock=pygame.time.Clock()

while not done:
    screen.fill(BLACK)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            done = True

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_LEFT:
    
                rect_change_x = -6
                check1 = True
    
            elif event.key == pygame.K_RIGHT:
    
                rect_change_x = 6
                check2 = True
                
            elif event.key == pygame.K_UP:
               check = True
            
            elif event.key == pygame.K_a:
                check3 = not check3
            elif event.key == pygame.K_b:
                check4 = not check4
            elif event.key == pygame.K_c:
                check5 = not check5
            elif event.key == pygame.K_d:
                startGame = not startGame

                     

        elif event.type == pygame.KEYUP:

            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:

                rect_change_x = 0
                check1 = False
                check2 = False

            elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:

                rect_change_y = 0  
                check = False


    rect_x += rect_change_x

    rect_y += rect_change_y

    

    ball_x += ball_change_x

    ball_y += ball_change_y

    
#author---Katha Roy


    if ball_x<0:

        ball_x=0

        ball_change_x = ball_change_x * -1

    elif ball_x>760:

        ball_x=760

        ball_change_x = ball_change_x * -1

    elif ball_y<0:

        ball_y=0

        ball_change_y = ball_change_y * -1

    elif ball_x>rect_x and ball_x<rect_x+100 and ball_y==565:

        ball_change_y = ball_change_y * -1

        score = score + 1

    elif ball_y>600:

        ball_change_y = ball_change_y * -1

        score = 0                        
    
    if(startGame):
        pygame.draw.rect(screen,WHITE,[ball_x,ball_y,15,15])

    

   
    drawrect(screen,rect_x,rect_y)
    if(check and not startGame):
        font= pygame.font.SysFont('Calibri', 30, True, False)
        text = font.render("Blink", True, WHITE)
        screen.blit(text,[350,300])
        
    if(check1 and not startGame):
        font= pygame.font.SysFont('Calibri', 30, True, False)
        text = font.render("Look Left", True, WHITE)
        screen.blit(text,[350,300])
        
    if(check2 and not startGame):
        font= pygame.font.SysFont('Calibri', 30, True, False)
        text = font.render("Look Right", True, WHITE)
        screen.blit(text,[350,300])
        
    if(check3):
        font= pygame.font.SysFont('Calibri', 30, True, False)
        text = font.render("Preparing classifier...", True, WHITE)
        screen.blit(text,[300,300])
        
    if(check4):
        font= pygame.font.SysFont('Calibri', 30, True, False)
        text = font.render("Classifier prepared!", True, WHITE)
        screen.blit(text,[300,300])
        
    if(check5):
        font= pygame.font.SysFont('Calibri', 30, True, False)
        text = font.render("Game starting...", True, WHITE)
        screen.blit(text,[300,300])

    if(startGame):
        font= pygame.font.SysFont('Calibri', 20, True, False)
        text = font.render("Score: " + str(score), True, WHITE)
        screen.blit(text,[600,100])    

       

    pygame.display.flip()         

    clock.tick(60)

    

pygame.quit()   

#author---Katha Roy