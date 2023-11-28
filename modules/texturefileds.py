import torch
import tinycudann as tcnn

class TextureFileds(torch.nn.Module):
    def __init__(self, n_input_dims, n_output_dims):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        #self.encoder = tcnn.
        #self.color_net = MLP(dim_in=dim_in, dim_hidden=dim_hidden, num_layers=num_layers, dim_out=dim_out)
        #self.act = torch.sigmoid
        config = {
            'encoding': {
                "otype": "Grid",
                "type": "Hash",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16, 
                "per_level_scale": 1.26, 
                "interpolation": "Smoothstep",

            },
            # "network": {
            #     "otype": "FullyFusedMLP",
            #     "activation": "ReLU",
            #     "output_activation": "None",
            #     "n_neurons": 64,  
            #     "n_hidden_layers": 5, 

            # }
        }
        # self.model = tcnn.NetworkWithInputEncoding(
	    #     n_input_dims, n_output_dims,
	    #     config["encoding"], config["network"])
        self.encoder = tcnn.Encoding(n_input_dims, config["encoding"])
        self.network = torch.nn.Sequential(
                    torch.nn.Linear(32, 64, bias=False),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 3, bias=False),
                    # torch.nn.ReLU(),
                    # torch.nn.Linear(16, 3, bias=False),
                    )
        # self.model = torch.nn.Sequential(
        #                 self.encoder,
        #                 self.network,
        # )
    def forward(self, x, latent_mode=False):
        x = self.encoder(x).to(torch.float)
        x = self.network(x)
        res = {'color': x.clamp(0, 1)}
        return res