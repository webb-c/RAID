from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Manager():
    def __init__(self, model_name='mobilenet', attack='PGD_Linf'):
        # main에서 선언 필요
        now = datetime.now()
        time = now.strftime('%m-%d_%H:%M:%S')
        self.writer = SummaryWriter(f'./logs/{model_name}_{attack}/{time}')

    def record(self, text, value, step):
        self.writer.add_scalar(text, value, step)