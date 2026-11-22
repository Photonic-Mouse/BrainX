import torch
import torch.nn.functional as F


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(
            self.save_activation
        )

    def save_activation(self, module, input, output):

        self.activations = output

        output.register_hook(self.save_gradient)

    def save_gradient(self, gradient):

        self.gradients = gradient

    def generate(self, image, target_class=None):

        self.model.zero_grad(set_to_none=True)

        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[:, target_class]

        score.backward()

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        cam = cam - cam.min()

        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), output.detach()
