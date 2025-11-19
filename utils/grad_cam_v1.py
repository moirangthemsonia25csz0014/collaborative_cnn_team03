
import os
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from torchvision import transforms


def locate_final_conv_layer(network: torch.nn.Module) -> torch.nn.Module:
    """Identify the last convolutional layer in the network architecture."""
    target_layer = None
    for layer in network.modules():
        if isinstance(layer, torch.nn.Conv2d):
            target_layer = layer
    if target_layer is None:
        raise RuntimeError("Unable to locate any Conv2d layer in the network.")
    return target_layer



def attach_activation_tensor_hook(conv_module: torch.nn.Module, feature_storage: dict, gradient_storage: dict):

    handles = []

    def forward_hook(module, inp, out):
        
        feature_storage['features'] = out.detach()

        def tensor_grad_hook(grad):
            gradient_storage['gradients'] = grad.detach()
        th = out.register_hook(tensor_grad_hook)
        handles.append(th)

    fh = conv_module.register_forward_hook(forward_hook)
    handles.insert(0, fh)
    return handles


def compute_activation_map(network: torch.nn.Module, input_data: torch.Tensor,
                           class_index: Optional[int] = None,
                           computation_device: Optional[str] = None) -> Tuple[np.ndarray, int]:

    network.eval()

    if computation_device is None:
        computation_device = next(network.parameters()).device
    else:
        computation_device = torch.device(computation_device)

    input_data = input_data.to(computation_device)

    conv_layer = locate_final_conv_layer(network)
    feature_store = {}
    gradient_store = {}

    handles = attach_activation_tensor_hook(conv_layer, feature_store, gradient_store)

    model_output = network(input_data)

    if class_index is None:
        class_index = int(model_output.argmax(dim=1).item())

    target_score = model_output[:, class_index]
    network.zero_grad()
    target_score.backward(retain_graph=True)

    captured_features = feature_store.get('features')
    captured_gradients = gradient_store.get('gradients')

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    if captured_features is None or captured_gradients is None:
        raise RuntimeError("Hook capture failed during Grad-CAM computation.")

    channel_weights = torch.mean(captured_gradients, dim=(2, 3), keepdim=True)
    weighted_activation = channel_weights * captured_features
    activation_map = torch.sum(weighted_activation, dim=1).squeeze(0)

    activation_map = F.relu(activation_map)
    activation_map = activation_map - activation_map.min()
    if activation_map.max() != 0:
        activation_map = activation_map / activation_map.max()

    heatmap_array = activation_map.cpu().numpy()
    return heatmap_array, int(class_index)


def apply_colormap_matplotlib(heatmap_data: np.ndarray, color_scheme: str = 'jet') -> np.ndarray:
    color_mapper = cm.get_cmap(color_scheme)
    colored_map = color_mapper(heatmap_data)
    rgb_output = np.uint8(colored_map[:, :, :3] * 255)
    return rgb_output


def blend_heatmap_with_image(source_image: Image.Image, activation_heatmap: np.ndarray, 
                             transparency: float = 0.45, color_scheme: str = 'jet') -> Image.Image:
    img_array = np.array(source_image.convert('RGB'))
    height, width = img_array.shape[:2]

    heatmap_pil = Image.fromarray(np.uint8(255 * activation_heatmap))
    resized_heatmap = heatmap_pil.resize((width, height), resample=Image.BILINEAR)
    normalized_heatmap = np.array(resized_heatmap).astype(np.float32) / 255.0

    colored_heatmap = apply_colormap_matplotlib(normalized_heatmap, color_scheme)

    heatmap_layer = colored_heatmap.astype(np.float32)
    image_layer = img_array.astype(np.float32)
    blended_output = image_layer * (1 - transparency) + heatmap_layer * transparency
    blended_output = np.uint8(np.clip(blended_output, 0, 255))
    
    return Image.fromarray(blended_output)


def visualize_single_image(network: torch.nn.Module, source_image: Image.Image, 
                          preprocessing: transforms.Compose,
                          computation_device: Optional[str] = None, 
                          class_index: Optional[int] = None,
                          transparency: float = 0.45, 
                          color_scheme: str = 'jet') -> Tuple[Image.Image, np.ndarray, int]:
    
    network.eval()
    
    if computation_device is None:
        computation_device = next(network.parameters()).device
    else:
        computation_device = torch.device(computation_device)

    tensor_input = preprocessing(source_image).unsqueeze(0).to(computation_device)
    heatmap, predicted_class = compute_activation_map(network, tensor_input, 
                                                      class_index=class_index, 
                                                      computation_device=computation_device)
    blended_image = blend_heatmap_with_image(source_image, heatmap, 
                                            transparency=transparency, 
                                            color_scheme=color_scheme)
    
    return blended_image, heatmap, predicted_class


def export_batch_visualizations(network: torch.nn.Module, data_loader, category_names, 
                                preprocessing: transforms.Compose,
                                output_directory: str = "../results/gradcam", 
                                computation_device: Optional[str] = None,
                                sample_count: int = 12, 
                                start_position: int = 0, 
                                color_scheme: str = 'jet'):
  
    os.makedirs(output_directory, exist_ok=True)
    
    if computation_device is None:
        computation_device = next(network.parameters()).device
    else:
        computation_device = torch.device(computation_device)

    source_dataset = data_loader.dataset
    if not hasattr(source_dataset, 'samples'):
        raise RuntimeError('Dataset must expose .samples attribute (ImageFolder required)')

    processed = 0
    dataset_length = len(source_dataset.samples)
    position = start_position

    while processed < sample_count and position < dataset_length:
        image_path, true_label = source_dataset.samples[position]
        
        try:
            pil_image = Image.open(image_path).convert('RGB')
        except Exception:
            position += 1
            continue

        blended, heatmap, prediction = visualize_single_image(
            network, pil_image, preprocessing=preprocessing, 
            computation_device=computation_device, transparency=0.45, 
            color_scheme=color_scheme
        )

        file_base = os.path.splitext(os.path.basename(image_path))[0]
        sanitized_class = category_names[true_label].replace(' ', '_')
        
        overlay_filename = os.path.join(
            output_directory, 
            f"{processed+1:03d}_{sanitized_class}_true{true_label}_pred{prediction}_{file_base}_overlay.png"
        )
        heatmap_filename = os.path.join(
            output_directory, 
            f"{processed+1:03d}_{sanitized_class}_true{true_label}_pred{prediction}_{file_base}_heat.png"
        )

        blended.save(overlay_filename)
        plt.imsave(heatmap_filename, heatmap, cmap=color_scheme)

        processed += 1
        position += 1

    print(f"Successfully saved {processed} Grad-CAM visualizations to {output_directory}")