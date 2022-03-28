# ALL relu
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, byte):
        super(AutoEncoder, self).__init__()
        if byte == 256:
            self.encoder = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
            self.decoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                # nn.ReLU()
                nn.ReLU()
            )

        elif byte == 128:
            self.encoder = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            self.decoder = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                # nn.ReLU()
                nn.ReLU()
            )

        elif byte == 64:
            self.encoder = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.decoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                # nn.ReLU()
                nn.ReLU()
            )

        else:
            self.encoder = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, byte),
            )
            self.decoder = nn.Sequential(
                nn.Linear(byte, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                # nn.ReLU()
                nn.ReLU()
            )

    def forward(self, input, tag=False):
        if tag == False:
            compress = self.encoder(input)
        else:
            compress = input
        reconstruct = self.decoder(compress)
        return compress, reconstruct
