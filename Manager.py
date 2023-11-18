from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision import transforms
import torch, os


class Manager():
    def __init__(self, model_name='mobilenet', attack='PGD_Linf', use=True):
        # main에서 선언 필요
        self.now = datetime.now()
        self.time = self.now.strftime('%m-%d_%H%M%S')
        self.use = use
        self.log_path = f'./logs/{model_name}_{attack}/{self.time}'
        self.img_save_path = f'./saved_images/{model_name}_{attack}/{self.time}'
        self.transform = transforms.ToPILImage()
        if self.use : 
            self.writer = SummaryWriter(self.log_path)

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