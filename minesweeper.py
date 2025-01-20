import pygame
import numpy as np
import math
import torch
import torch.nn.functional as F

from collections import deque
# hver eneste frame, opdater new_state
# tilføj SARST til deque
# opdater state
# I så har tilføjet framet INDEN til replay bufferen

#buffer = deque(len=10000)


class MineSweeper():
    rendering=False
    #Load mine
    file = 'mine.png'
    images = pygame.image.load(file)

    def __init__(self,size,state=None):
        pygame.init()
        if state is None:
            self.size=size
            self.board,self.output=self.reset()
        self.reward = 0
        self.did_click_again = 0 

    def board_creater(self,first_choice):
        #creates board based on the first choice
        y_choice,x_choice = first_choice
        mine_num=np.random.randint(int(self.size**2/4),int(self.size**2/3)) # antal miner
        solution=np.zeros((self.size,self.size),dtype=int)
        ## reserverer indgangsstedet, så der ikke kan placeres miner
        for x_ in range (x_choice-1,x_choice+2):
            for y_ in range(y_choice-1,y_choice+2):
                if 0<=x_<self.size and 0<=y_<self.size:
                    solution[y_,x_]=-2
        self.numbomb = 0
        ## placerer minerne tilfældigt, men ikke oveni 
        # hinanden eller indgangsstedet
        for i in range (mine_num):
            x_bomb,y_bomb=np.random.randint(0,self.size),np.random.randint(0,self.size)
            if solution[y_bomb,x_bomb]==0:
                solution[y_bomb, x_bomb]= -1
                self.numbomb += 1
        # Laver counter for hvor mange bomber er rundt om feldtet 
        for x in range (self.size):
            for y in range(self.size):
                if solution[y,x]!=-1:
                    counter=0
                    for x_small in range(x-1,x+2):
                        for y_small in range(y-1,y+2):
                            if self.size>x_small>=0 and self.size>y_small>=0:
                                if solution[y_small,x_small]==-1:
                                    counter+=1
                    solution[y,x]=counter
        return solution        
    
    def reset(self):
            
        #### laver indledende input og ouput-skabelon
        self.board=np.ones((self.size,self.size),dtype=int)*9
        self.output=np.zeros((self.size,self.size),dtype=int)
        self.game_over = False
        self.won=False
        self.new_game = True
        return self.board,self.output
    
    def get_input(self):
        ## tjekker løsing og opdaterer boardet
        for y in range(self.size):
            for x in range(self.size): 
                if self.output[y,x]==1 and self.board[y,x]==9:  
                    self.output[y,x]=0
                    self.board[y,x]=self.solution[y,x]
                    self.reward += 4 #reward for valid guess
                    if self.solution[y,x]==-1:  # hvis valgte er bombe
                        self.reward = -100 #reward for bomb
                        self.game_over=True
                    ## hvis den valgte tile er nul:
                    elif self.board[y,x]==0:
                        self.output=self.zero_slot(self.output,y,x)
                        self.get_input()
                elif self.output[y,x] == 1 and self.board[y,x]!=9:
                    self.reward = -250
                    self.did_click_again += 1 
        return self.reward
        
                
    def step(self, y, x):
        self.reward = 0
        state=self.board.copy()
        self.choose(x,y)
        self.action = np.array([y,x])
        self.did_win()
        return (state, self.action, self.reward, self.board, self.game_over)

    def did_win(self):
        counter=0
        for x in range(self.size):
            for y in range(self.size):
                if self.board[y,x] == 9:
                    counter += 1
        if counter == self.numbomb:
            self.reward = 100
            self.won = True
            self.game_over = True
        return self.reward, self.did_click_again


    def choose(self,x,y):
        if self.new_game == True:
            # Creates the solution-board after the first click - ensuring first click is not a bomb
            self.solution = self.board_creater((y,x))
            self.new_game = False
        self.output[y,x] = 1
        self.get_input()
        return self.reward

    def zero_slot(self,output,y,x):
        ## if the chosen slot has value zero, 
        ## choose the sorrounding 8 tiles
        for x_small in range(x-1,x+2):
            for y_small in range(y-1,y+2):
                if self.size>x_small>=0 and self.size>y_small>=0:
                    if self.board[y_small,x_small]==9:
                        output[y_small,x_small]=1
        return output

    def close(self):
        pygame.quit()

    def init_render(self):
        ## set up the screen for rendering
        self.screen=pygame.display.set_mode([600,600])
        pygame.display.set_caption("Minesweeper")
        self.background=pygame.Surface(self.screen.get_size())
        self.rendering=True
        self.clock=pygame.time.Clock()
        self.font = pygame.font.Font(None, 25)

    def render(self):
        if not self.rendering:
            self.init_render()
        
        # Limit to 30 fps
        self.clock.tick(15)

        # Clear the screen
        self.screen.fill((128,128,128))

        # Draw board
        blank_color=(220,220,220)
        revealed_color=(192,192,192)

        for y in range(self.size):
            for x in range(self.size):
                if self.board[y,x]==9 or self.board[y,x] == 11:
                    color=blank_color
                    text=self.font.render("",True,(0,0,0))
                else:
                    color=revealed_color
                    if 1<=self.board[y,x]<=2:
                        text=self.font.render(str(self.board[y,x]),True,(220,20,60))
                    elif 3<=self.board[y,x]<=4:
                        text=self.font.render(str(self.board[y,x]),True,(233,116,81))
                    elif 5<=self.board[y,x]<=8:
                        text=self.font.render(str(self.board[y,x]),True,(253,218,13))
                    elif self.board[y,x]==-1:  
                        ### indsæt bombe
                        self.bombimage = self.images
                        self.bombimage = pygame.transform.scale(self.bombimage,(600/self.size-3,600/self.size-3))
                        text = self.bombimage
                    else:
                        text=self.font.render("",True,(220,20,60))
                
                rect=pygame.Rect(x*(600/self.size),y*(600/self.size),600/self.size-3,600/self.size-3)
                pygame.draw.rect(self.screen,color,rect)
                ## finder midten af firkanten
                text_rect=text.get_rect(center=rect.center)
                # tegner teksten
                self.screen.blit(text,text_rect)

        if self.game_over:
            if self.won:
                msg="Congratualtions!"
                color=(0,255,0)
            else:
                msg="You lost:("
                color=(255,0,0)
            self.game_overfont = pygame.font.Font(None,75) 
            text=self.game_overfont.render(msg,True,color)
            textpos=text.get_rect(centerx=self.background.get_width()/2)
            textpos.top=200
            self.screen.blit(text,textpos)

        ## tegner det hele
        pygame.display.flip()
    
    def one_hot_encode(self,state):
        one_hot = np.zeros((10, self.size,self.size))
        for z in range(10):
            for y in range(self.size):
                for x in range(self.size):
                    if state[y,x] == z:
                        one_hot[z,y,x] = 1
                    else:
                        one_hot[z,y,x] = 0
        hot=torch.tensor(one_hot, dtype= torch.float32)
        return hot
    
           
            



