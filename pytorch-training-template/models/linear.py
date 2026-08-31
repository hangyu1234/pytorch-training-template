import torch.nn as nn


class LinearRegressionModel(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()
        self.linear = nn.Linear(
            input_dim,
            output_dim,
        )

    def forward(self, x):
        prediction = self.linear(x)
        return prediction


if __name__ == "__main__":

    import torch


    model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )


    x = torch.randn(
        32,
        3
    )


    y = model(x)


    print("Input shape:")
    print(x.shape)


    print("Output shape:")
    print(y.shape)


    print("Parameters:")

    for name, param in model.named_parameters():
        print(name, param.shape)