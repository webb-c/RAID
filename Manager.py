from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision import transforms
import torch, os


class Manager():
    def __init__(self, model_name='mobilenet', attack='PGD_Linf', use=True, info="", action=False):
        # main에서 선언 필요
        if info!="":
            info += '/'
        self.now = datetime.now()
        self.time = self.now.strftime('%m-%d_%H%M%S')
        self.use = use
        self.log_path = f'./logs/{model_name}_{attack}/{info}{self.time}'
        self.img_save_path = f'./saved_images/{model_name}_{attack}/{info}{self.time}'
        self.action_save_path = f'./action_log/{model_name}_{attack}/{info}{self.time}.txt'
        self.transform = transforms.ToPILImage()
        self.action = action
        if self.use : 
            self.writer = SummaryWriter(self.log_path)
        if self.action :
            f = open(self.action_save_path, 'w')
            f.close()

    def record(self, text, value, step):
        if self.use : 
            self.writer.add_scalar(text, value, step)
        
    def get_time(self):
        return self.now
    
    def get_log_path(self):
        return self.log_path
    
    def save_image(self, episode, step, img):
        os.makedirs(f"{self.img_save_path}/episode_{episode}/", exist_ok=True)
        path = f"{self.img_save_path}/episode_{episode}/step_{step}.png"
        self.transform(torch.tensor(img)).save(path)

    def save_action(self, episode, step, actions, epi=True):
        f = open(self.action_save_path, 'a')  
        if epi:
            # write episode
            f.write(f'\nepisode {episode}\n')
        else:
            # write step actions
            text = 'step {:d}\t'.format(episode, step)
            action, log_prob = "", ""
            for idx in range(actions)//2:
                action += f'{actions[idx]}, '
                log_prob += f'{actions[len(actions)//2+idx]}, '
            text = text + action[:-2]+'\t'+actions[:-2]
            f.write(text)
        f.close()