import torch
from networks import DenseNet121Heat, LVNet


class LandmarkDetection(object):
    def __init__(self, model_name, kpt):
        if model_name == 'DenseNet121Heat':
            model = DenseNet121Heat()
        elif model_name == 'LVNet':
            model = LVNet()

        self.model_name = model_name
        self.model = model.load_state_dict(torch.load(kpt))

    def estimate(self, im_tensor):
        model = self.model
        output = model(im_tensor)
        return output


