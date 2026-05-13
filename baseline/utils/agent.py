#import torchvision.datasets.SBDataset as sbd
from utils.models import *
from utils.tools import *
import os
import imageio
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
from torch.autograd import Variable

from tqdm.notebook import tqdm
# Removed config import since it's missing, will define inline
# from config import *
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
criterion = nn.SmoothL1Loss()
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
Tensor = torch.FloatTensor

import glob
from PIL import Image

class Agent():
    def __init__(self, classe="mixing", alpha=0.1, nu=3.0, threshold=0.5, max_steps=20, load=False, device='cpu', extractor_name='vgg16', use_cache=True):
        self.BATCH_SIZE = 100
        self.GAMMA = 0.900
        self.EPS = 1  #epsilon 
        self.TARGET_UPDATE = 1 
        self.save_path = "./models/q_network"
        screen_height, screen_width = 224, 224
        self.n_actions = 9
        self.classe = classe
        self.device = device
        self.use_cuda = (str(self.device) != 'cpu')


        self.feature_extractor = get_backbone(extractor_name)
        input_dim = self.feature_extractor.output_dim + 81 # 81 is history_dim (9*9)

        if not load:
            self.policy_net = DQN(input_dim, self.n_actions)
        else:
            self.policy_net = self.load_network()
            
        self.target_net = DQN(input_dim, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.feature_extractor.eval()
        if self.use_cuda:
          self.feature_extractor = self.feature_extractor.cuda()
          self.target_net = self.target_net.cuda()
          self.policy_net = self.policy_net.cuda()
        
        # Compatibility wrapper for v2 pipeline (accepts img_feat, hist_feat)
        class ModelWrapper(nn.Module):
            def __init__(self, policy_net):
                super().__init__()
                self.policy_net = policy_net
            def forward(self, img_feat, hist_feat=None):
                # If hist_feat is None, we assume img_feat is already concatenated (fallback)
                if hist_feat is None:
                    return self.policy_net(img_feat)
                # Ensure they are batched correctly
                if img_feat.dim() == 1: img_feat = img_feat.unsqueeze(0)
                if hist_feat.dim() == 1: hist_feat = hist_feat.unsqueeze(0)
                # Concatenate along feature dimension
                x = torch.cat((img_feat, hist_feat), 1)
                return self.policy_net(x)
        
        self.model = ModelWrapper(self.policy_net)
        
        # v2 Compatibility attributes
        from v2.backbone.engine import DQNEngine
        self.engine = DQNEngine(self.model, gamma=self.GAMMA, use_target_net=True)
        self.engine.target_model = ModelWrapper(self.target_net) # Wrapper for target net too
        self.loss_fn = criterion

        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        
        self.alpha = alpha # €[0, 1]  Scaling factor
        self.nu = nu # Reward of Trigger
        self.threshold = threshold
        self.max_steps = max_steps
        self.history_size = 9
        self.actions_history = torch.zeros((9, 9))
        if self.use_cuda:
            self.actions_history = self.actions_history.cuda()
        self.num_episodes = 15
        
        self.use_cache = use_cache
        self.last_next_state = None
        self.last_mask = None

    def save_network(self):
        torch.save(self.policy_net, self.save_path+"_"+self.classe)
        print('Saved')

    def load_network(self):
        if not self.use_cuda:
            return torch.load(self.save_path+"_"+self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path+"_"+self.classe)



    def intersection_over_union(self, box1, box2):
        x11, y11, x21, y21 = box1
        x12, y12, x22, y22 = box2

        xi1 = max(x11, x12)
        yi1 = max(y11, y12)
        xi2 = min(x21, x22)
        yi2 = min(y21, y22)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou

    def compute_iou(self, mask, ground_truth):
        return self.intersection_over_union(mask, ground_truth)

    def compute_reward(self, action, current_mask, ground_truth):
        # Handle both (action, current_mask, ground_truth) and (new_mask, prev_mask, ground_truth)
        if isinstance(action, (int, np.integer)):
            new_mask = self.compute_mask(action, current_mask)
            iou_new = self.compute_iou(new_mask, ground_truth)
            iou_current = self.compute_iou(current_mask, ground_truth)
        else:
            # Backward compatibility for baseline internal calls
            new_mask = action
            iou_new = self.compute_iou(new_mask, ground_truth)
            iou_current = self.compute_iou(current_mask, ground_truth)
        
        # Strictly integer-based rewards
        if iou_new > iou_current:
            # Strict logic: Agent ONLY gets positive reward if it matches ground truth (> threshold)
            if iou_new >= self.threshold:
                return 1.0
            else:
                return 0.0
        else:
            return -1.0
      
    def rewrap(self, coord):
        return min(max(coord,0), 224)
      
    def compute_finish_reward(self, current_mask, ground_truth):
        iou = self.compute_iou(current_mask, ground_truth)
        if iou >= self.threshold:
            # Scale the exponential IoU up to allow meaningful integers, then strictly round down
            reward = self.nu * (iou ** 2) * 10.0
            return float(int(reward))
        else:
            return -float(int(self.nu))

    def compute_trigger_reward(self, actual_state, ground_truth):
        # Legacy name for compute_finish_reward
        return self.compute_finish_reward(actual_state, ground_truth)

    def get_best_next_action(self, actions, ground_truth):
        positive_actions = []
        negative_actions = []
        actual_equivalent_coord = self.calculate_position_box(actions)
        for i in range(0, 9):
            copy_actions = actions.copy()
            copy_actions.append(i)
            new_equivalent_coord = self.calculate_position_box(copy_actions)
            if i != 8: # Action 8 is trigger in new mapping
                reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, ground_truth)
            else:
                reward = self.compute_finish_reward(new_equivalent_coord, ground_truth)
            
            if reward >= 0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions) == 0:
            return random.choice(negative_actions)
        return random.choice(positive_actions)


    def select_action(self, state, actions, ground_truth):
        sample = random.random()
        eps_threshold = self.EPS
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                if self.use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] # + 1
                try:
                  return action.cpu().numpy()[0]
                except:
                  return action.cpu().numpy()
        else:
            return self.get_best_next_action(actions, ground_truth)

    def select_action_model(self, state):
        with torch.no_grad():
                if self.use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] # + 1
                return action

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_step(self, batch_size=20):
        self.BATCH_SIZE = batch_size
        return self.optimize_model()

    def step(self, image, history, current_mask, ground_truth, step_count, epsilon):
        self.EPS = epsilon
        
        # Crop image and resize to 224x224
        import cv2
        x1, y1, x2, y2 = map(int, current_mask)
        cropped_img = image[y1:y2, x1:x2]
        if cropped_img.size == 0:
            cropped_img = image
            
        cropped_img = cv2.resize(cropped_img, (224, 224))
        img_transposed = np.transpose(cropped_img, (2, 0, 1))
        img_tensor = torch.from_numpy(img_transposed).float() / 255.0
        
        # Reconstruct actions history for baseline API
        self.actions_history = torch.zeros((9, 9))
        size_h = sum([1 for h in history if h != -1])
        valid_hist = [h for h in history if h != -1]
        for i, h in enumerate(reversed(valid_hist)):
            if i < 9:
                self.actions_history[i][h] = 1
                
                
        # Feature Caching Logic
        if self.use_cache and self.last_next_state is not None and np.array_equal(current_mask, self.last_mask):
            state = self.last_next_state
        else:
            state = self.compose_state(img_tensor, dtype=torch.FloatTensor)
            if self.use_cuda:
                state = state.cuda()
        
        if step_count >= self.max_steps:
            action = 8 # Action 8 is trigger in new mapping
        else:
            action = self.select_action(state, valid_hist, ground_truth)
            
        if action == 8: # Action 8 is trigger
            new_mask = current_mask
            reward = self.compute_finish_reward(current_mask, ground_truth)
            done = True
        else:
            self.actions_history = self.update_history(action)
            new_mask = self.compute_mask(action, current_mask)
            reward = self.compute_reward(new_mask, current_mask, ground_truth)
            done = False
            
        if done:
            next_state = None
        else:
            nx1, ny1, nx2, ny2 = map(int, new_mask)
            next_cropped = image[ny1:ny2, nx1:nx2]
            if next_cropped.size == 0:
                next_cropped = image
            next_cropped = cv2.resize(next_cropped, (224, 224))
            next_transposed = np.transpose(next_cropped, (2, 0, 1))
            next_img_tensor = torch.from_numpy(next_transposed).float() / 255.0
            next_state = self.compose_state(next_img_tensor, dtype=torch.FloatTensor)
            if self.use_cuda:
                next_state = next_state.cuda()
            
            # Update cache for next step
            if self.use_cache:
                self.last_next_state = next_state
                self.last_mask = new_mask
            
        self.memory.push(state.cpu(), int(action), next_state.cpu() if next_state is not None else None, reward)
        
        # Update history array
        history = history[1:] + [action]
        return new_mask, reward, done, history

    def compute_mask(self, action, current_mask):
        return self.calculate_position_box_v2(action, current_mask)

    def calculate_position_box_v2(self, action, current_mask):
        alpha_w = self.alpha * (current_mask[2] - current_mask[0])
        alpha_h = self.alpha * (current_mask[3] - current_mask[1])
        real_x_min, real_y_min, real_x_max, real_y_max = current_mask
        r = action
        if r == 0: # Right
            real_x_min += alpha_w; real_x_max += alpha_w
        elif r == 1: # Left
            real_x_min -= alpha_w; real_x_max -= alpha_w
        elif r == 2: # Down
            real_y_min += alpha_h; real_y_max += alpha_h
        elif r == 3: # Up
            real_y_min -= alpha_h; real_y_max -= alpha_h
        elif r == 4: # Bigger
            real_y_min -= alpha_h; real_y_max += alpha_h
            real_x_min -= alpha_w; real_x_max += alpha_w
        elif r == 5: # Smaller
            real_y_min += alpha_h; real_y_max -= alpha_h
            real_x_min += alpha_w; real_x_max -= alpha_w
        elif r == 6: # Taller (Y shrinking)
            real_y_min += alpha_h; real_y_max -= alpha_h
        elif r == 7: # Fatter (X shrinking)
            real_x_min += alpha_w; real_x_max -= alpha_w
        elif r == 8: # Trigger
            return current_mask
            
        new_mask_tmp = np.array([real_x_min, real_y_min, real_x_max, real_y_max])
        new_mask = np.array([
            min(new_mask_tmp[0], new_mask_tmp[2]),
            min(new_mask_tmp[1], new_mask_tmp[3]),
            max(new_mask_tmp[0], new_mask_tmp[2]),
            max(new_mask_tmp[1], new_mask_tmp[3])
        ])
        return new_mask

    def optimize_model(self):
 

        if len(self.memory) < self.BATCH_SIZE:
            return 0.0
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool()
        next_states = [s for s in batch.next_state if s is not None]
        
        # Use torch.no_grad() instead of volatile
        with torch.no_grad():
            non_final_next_states = torch.cat(next_states).type(Tensor)
            if self.use_cuda:
                non_final_next_states = non_final_next_states.cuda()
        
        state_batch = torch.cat(batch.state).type(Tensor)
        action_batch = torch.LongTensor(batch.action).view(-1,1).type(LongTensor)
        reward_batch = torch.FloatTensor(batch.reward).view(-1,1).type(Tensor)

        if self.use_cuda:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, 1).type(Tensor)
        if self.use_cuda:
            next_state_values = next_state_values.cuda()

        with torch.no_grad():
            if non_final_mask.any():
                d = self.target_net(non_final_next_states) 
                next_state_values[non_final_mask] = d.max(1)[0].view(-1,1)

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    
    def feature_extract(self, image, history, width, height, current_mask):
        import cv2
        x1, y1, x2, y2 = map(int, current_mask)
        # Ensure coordinates are within bounds
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        
        cropped_img = image[y1:y2, x1:x2]
        if cropped_img.size == 0:
            cropped_img = image
            
        cropped_img = cv2.resize(cropped_img, (224, 224))
        img_transposed = np.transpose(cropped_img, (2, 0, 1))
        img_tensor = torch.from_numpy(img_transposed).float() / 255.0
        if self.use_cuda:
            img_tensor = img_tensor.cuda()
            
        feature_tensor = self.get_features(img_tensor.unsqueeze(0), dtype=torch.FloatTensor)
        
        # Reconstruct actions history for baseline API
        history_tensor = torch.zeros((1, 81)) # 9*9
        valid_hist = [h for h in history if h != -1]
        for i, h in enumerate(reversed(valid_hist)):
            if i < 9:
                history_tensor[0][i*9 + h] = 1
        
        if self.use_cuda:
            history_tensor = history_tensor.cuda()
            
        return feature_tensor, history_tensor

    def compose_state(self, image, dtype=FloatTensor):
        image_feature = self.get_features(image, dtype)
        image_feature = image_feature.view(1,-1)
        # Ensure all tensors are on the same device
        history_flatten = self.actions_history.view(1,-1).type(dtype).to(image_feature.device)
        state = torch.cat((image_feature, history_flatten), 1)
        return state
    
    def get_features(self, image, dtype=FloatTensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.type(dtype)
        if self.use_cuda:
            image = image.cuda()
        feature = self.feature_extractor(image)
        return feature.data

    
    def update_history(self, action):
        action_vector = torch.zeros(9).to(self.actions_history.device)
        action_vector[action] = 1
        size_history_vector = len(torch.nonzero(self.actions_history))
        if size_history_vector < 9:
            self.actions_history[size_history_vector][action] = 1
        else:
            for i in range(8,0,-1):
                self.actions_history[i][:] = self.actions_history[i-1][:]
            self.actions_history[0][:] = action_vector[:] 
        return self.actions_history

    def calculate_position_box(self, actions, xmin=0, xmax=224, ymin=0, ymax=224):
        alpha_h = self.alpha * (  ymax - ymin )
        alpha_w = self.alpha * (  xmax - xmin )
        real_x_min, real_y_min, real_x_max, real_y_max = 0, 0, 224, 224

        for r in actions:
            if r == 0: # Right
                real_x_min += alpha_w; real_x_max += alpha_w
            elif r == 1: # Left
                real_x_min -= alpha_w; real_x_max -= alpha_w
            elif r == 2: # Down
                real_y_min += alpha_h; real_y_max += alpha_h
            elif r == 3: # Up
                real_y_min -= alpha_h; real_y_max -= alpha_h
            elif r == 4: # Bigger
                real_y_min -= alpha_h; real_y_max += alpha_h
                real_x_min -= alpha_w; real_x_max += alpha_w
            elif r == 5: # Smaller
                real_y_min += alpha_h; real_y_max -= alpha_h
                real_x_min += alpha_w; real_x_max -= alpha_w
            elif r == 6: # Taller (Y shrinking)
                real_y_min += alpha_h; real_y_max -= alpha_h
            elif r == 7: # Fatter (X shrinking)
                real_x_min += alpha_w; real_x_max -= alpha_w
        real_x_min, real_y_min, real_x_max, real_y_max = self.rewrap(real_x_min), self.rewrap(real_y_min), self.rewrap(real_x_max), self.rewrap(real_y_max)
        return [real_x_min, real_y_min, real_x_max, real_y_max]

    def get_max_bdbox(self, ground_truth_boxes, actual_coordinates ):
        max_iou = False
        max_gt = []
        for gt in ground_truth_boxes:
            iou = self.intersection_over_union(actual_coordinates, gt)
            if max_iou == False or max_iou < iou:
                max_iou = iou
                max_gt = gt
        return max_gt

    def predict_image(self, image, plot=False,choice  = 1 ):
        self.policy_net.eval()
        xmin = 0
        xmax = 224
        ymin = 0
        ymax = 224

        done = False
        all_actions = []
        self.actions_history = torch.ones((9,9))
        state = self.compose_state(image)
        original_image = image.clone()
        new_image = image

        steps = 0
        
        while not done:
            steps += 1
            action = self.select_action_model(state)
            all_actions.append(action)
            if action == 8: # Action 8 is trigger
                next_state = None
                new_equivalent_coord = self.calculate_position_box(all_actions)
                done = True
            else:
                self.actions_history = self.update_history(action)
                new_equivalent_coord = self.calculate_position_box(all_actions)            
                
                new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                try:
                    new_image = transform(new_image)
                except ValueError:
                    break            
                
                next_state = self.compose_state(new_image)
            
            if steps == 40:
                done = True
            
            state = next_state
            image = new_image
        
            if plot:
                show_new_bdbox(original_image, new_equivalent_coord, color='b', count=steps)
        

        if plot:
            if (choice) : 
                tested = 0
                while os.path.isfile('media/movie_'+str(tested)+'.gif'):
                    tested += 1
                fp_out = "media/movie_"+str(tested)+".gif"
                images = []
                for count in range(1, steps+1):
                    images.append(imageio.imread(str(count)+".png"))
                
                imageio.mimsave(fp_out, images)
                
                for count in range(1, steps):
                    os.remove(str(count)+".png")
            if (not choice) : 
                tested = 0
                while os.path.isfile('media2/movie_'+str(tested)+'.gif'):
                    tested += 1
                fp_out = "media2/movie_"+str(tested)+".gif"
                images = []
                for count in range(1, steps+1):
                    images.append(imageio.imread(str(count)+".png"))
                
                imageio.mimsave(fp_out, images)
                
                for count in range(1, steps):
                    os.remove(str(count)+".png")
        return new_equivalent_coord


    
    def evaluate(self, dataset):
        ground_truth_boxes = []
        predicted_boxes = []
        print("Predicting boxes...")
        for key, value in dataset.items():
            image, gt_boxes = extract(key, dataset)
            bbox = self.predict_image(image)
            ground_truth_boxes.append(gt_boxes)
            predicted_boxes.append(bbox)
        print("Computing recall and ap...")
        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes)
        print("Final result : \n"+str(stats))
        return stats
    
    def train_validate(self, train_loader, valid_loader,classe):
        op = open("logs_over_epochs", "a")
        op.write("class = "+str(classe))
        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0
        for i_episode in range(self.num_episodes):  
            print("Episode "+str(i_episode))
            for key, value in  train_loader.items():
                image, ground_truth_boxes = extract(key, train_loader)
                original_image = image.clone()
                ground_truth = ground_truth_boxes[0]
                all_actions = []
        
                # Initialize the environment and state
                self.actions_history = torch.ones((9,9))
                state = self.compose_state(image) # 81 + 25088 
                original_coordinates = [xmin, xmax, ymin, ymax]
                new_image = image
                done = False
                t = 0
                actual_equivalent_coord = original_coordinates
                new_equivalent_coord = original_coordinates
                while not done:
                    t += 1
                    action = self.select_action(state, all_actions, ground_truth)
                    all_actions.append(action)
                    if action == 8: # Action 8 is trigger
                        next_state = None
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_finish_reward(new_equivalent_coord,  closest_gt)
                        done = True

                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        

                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)
                        
                        actual_equivalent_coord = new_equivalent_coord
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)

                    # Vers le nouvel état
                    state = next_state
                    image = new_image
                    # Optimisation
                    self.optimize_model()
                    
            stats = self.evaluate(valid_loader)
            op.write("\n")
            op.write("Episode "+str(i_episode))
            op.write("\n")
            op.write(str(stats))
            op.write("\n")
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode<5:
                self.EPS -= 0.18
            self.save_network()
            
            print('Complete')