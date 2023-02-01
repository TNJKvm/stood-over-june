import torch.nn as nn



class MLP512(nn.Module):
    def __init__(self):
        super(MLP512, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(True),
            nn.Dropout(0),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0),
            nn.Linear(512, 10),
        )

    def forward(self, image):
        image = image.view(-1, 28 * 28)
        output = self.model(image)
        return output