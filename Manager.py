from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Manager():
    def __init__(self, model_name='mobilenet', attack='PGD_Linf', use=True):
        # main에서 선언 필요
        now = datetime.now()
        time = now.strftime('%m-%d_%H%M%S')
        self.use = use
        if self.use : 
            self.writer = SummaryWriter(f'./logs/{model_name}_{attack}/{time}')

    def record(self, text, value, step):
        if self.use : 
            self.writer.add_scalar(text, value, step)
        