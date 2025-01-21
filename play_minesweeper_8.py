import numpy as np 
import pygame
import math
from minesweeper import *
from ai_minesweep2 import *
import torch
from torch.utils.data import DataLoader, TensorDataset
device = "cuda" if torch.cuda.is_available() else "cpu"
env= MineSweeper(6)
model=Network2(6).to(device)

env.reset()

exit_program=False
runai = True
render = False
done = False

random=False 
clock = pygame.time.Clock()

## start program
while not exit_program:


    if render:
        env.render()
    if env.game_over == True:
        env.reset()
    if not runai:
    #game events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
            if event.type == pygame.MOUSEBUTTONDOWN:
                left, _, _= pygame.mouse.get_pressed()
                if left == 1:
                    x,y = pygame.mouse.get_pos()
                    x_pos = math.ceil(x/(600/env.size)) - 1
                    y_pos = math.ceil(y/(600/env.size)) - 1
                    state, action, reward, new_state, game_over = env.step(y_pos, x_pos)
    
    if runai:
        ### Logging data 
        wins = []
        p_vals =[]
        losses = []
        click_repeat = []
        click_count = 0
        episode_steps = []
        step_count = 0
        model.add_buffer()
        i=1
        while i<=iteration_max:
            render=False
            env.reset()
            episode_step = 0
            episode_loss = 0
            episode_gradient_step = 0
            # reduces exploration rate
            epsilon = (epsilon-epsilon_min)*epsilon_reduction_factor + epsilon_min
            while (not env.game_over) and (episode_step<max_episode_step):
                # Her spiller modellen og tilføjer til replay buffer
                state=env.one_hot_encode(env.board).to(device)
                # vælger action
                if np.random.rand()<epsilon:
                    # action vælges tilfældigt
                    y=np.random.randint(0,env.size)
                    x=np.random.randint(0,env.size)
                else:
                    array=np.array(model.forward(state).detach().clone()[0].unflatten(0,(env.size,env.size))) 
                    # og action vælges
                    y,x=np.unravel_index(np.argmax(array),array.shape)
       
                # der tages et step
                state,action,reward,new_state,is_terminal=env.step(y,x)
                step_count+=1
                episode_step+=1
          
                ## store to buffers
                state = env.one_hot_encode(state)
                new_state=env.one_hot_encode(new_state)
                buffer_index = step_count % buffer_len
                model.buff_state[buffer_index]=state
                model.buff_action[buffer_index]=torch.tensor(action,dtype=int)
                model.buff_reward[buffer_index]=reward
                model.buff_new_state[buffer_index]=new_state
                model.buff_is_terminal[buffer_index]=is_terminal
               
                # Learn with minibatch from buffer
                if step_count > batch_size and step_count%steps_per_gradient_update==0:
                    # Choose a minibatch            
                    batch_idx = np.random.choice(np.minimum(buffer_len, step_count), size=batch_size, replace=False)
                    # comupte loss function
                    out = model.forward(model.buff_state[batch_idx]).unflatten(1,(env.size,env.size)).to(device)
                    value = out[np.arange(batch_size), model.buff_action[batch_idx,0].long().to(device),model.buff_action[batch_idx,1].long().to(device)].flatten()
                    with torch.no_grad():
                        out_next=model.forward(model.buff_new_state[batch_idx].to(device))
                        target=(model.buff_reward[batch_idx].to(device) + gamma*(torch.max(out_next)) * (1-model.buff_is_terminal[batch_idx].to(device))).flatten()
                    loss=model.loss_function(value,target)
                    losses.append([loss.item()])
                    ## step the optimizer
                    model.update_weights(loss)
               
            if (i) % iteration_period == 0:
                #Tester modellen hver 10.000 spil 
                win=0
                for j in range(test_games):
                    env.reset()
                    episode_step=0
                    with torch.no_grad():
                        while (not env.game_over) and (episode_step < max_episode_step):
                            state=env.one_hot_encode(env.board)
                   
                            array=np.array(model.forward(state).detach().clone()[0].unflatten(0,(env.size,env.size)))
                            # og action vælges
                            y,x=np.unravel_index(np.argmax(array),array.shape)
                            # der tages et step
                            state,action,reward,new_state,is_terminal=env.step(y,x)
                            episode_step+=1
                            if env.won:
                                win+=1
                click_repeat.append(click_count)
                wins.append([win])
                p_vals.append([win/test_games])
                if (i )% 20000 == 0:
                    name=f"model6_{i/1000}K.pt"
                    torch.save(model.state_dict(), name)
                    
            episode_steps.append([episode_step])
            i+=1
    torch.save(model.state_dict(),'model6_400kfull.pt')

    import csv
    with open("losses_data6.csv","w",newline="") as file:
        writer = csv.writer(file)
        writer.writerows(losses)
    with open("steps_data6.csv","w",newline="") as file:
        writer = csv.writer(file)
        writer.writerows(episode_steps)
    with open("wins_test_data6.csv","w",newline="") as file:
        writer = csv.writer(file)
        writer.writerows(wins)
    with open("pvals_test_data6.csv","w",newline="") as file:
        writer = csv.writer(file)
        writer.writerows(p_vals)


    exit_program=True
              
               

        



env.close()


 
