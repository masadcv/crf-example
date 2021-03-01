import matplotlib.pyplot as plt
import nibabel
import torch

from monai.networks.blocks import CRF
from monai.utils.profiling import torch_profiler_time_end_to_end

@torch_profiler_time_end_to_end
def run_timed(func, args):
    return func(*args)

# Run parameters
run_in_3d = False
slice_index = 72

# Loading data from files
ct = nibabel.load("abdominal_ct.nii").get_fdata()
logits = nibabel.load("abdominal_logits.nii").get_fdata()

# Preparing data
if not run_in_3d:
    ct = ct[..., slice_index]
    logits = logits[..., slice_index]

ct_tensor = torch.from_numpy(ct).type(torch.FloatTensor).cuda()
logits_tensor = torch.from_numpy(logits).type(torch.FloatTensor).cuda()

# Adding noise
logits_tensor *= (torch.rand_like(logits_tensor) + 1) * 0.5

# CRF paramters
crf = CRF(
    iterations = 5, 
    bilateral_weight = 3.0,
    gaussian_weight = 1.0,
    bilateral_spatial_sigma = 5.0,
    bilateral_color_sigma = 0.5,
    gaussian_spatial_sigma = 5.0,
    compatability_kernel_range = 1,
)

# Run CRF and take labelsfrom input and output tensors
logits_smoothed_tensor = run_timed(crf, (logits_tensor, ct_tensor))
labels_tensor = torch.argmax(logits_tensor, dim=1, keepdim=True)
labels_smoothed_tensor = torch.argmax(logits_smoothed_tensor, dim=1, keepdim=True)

# Reading back data
ct = ct_tensor.squeeze(0).movedim(0, -1).cpu()
labels = labels_tensor.squeeze(0).movedim(0, -1).cpu()
labels_smoothed = labels_smoothed_tensor.squeeze(0).movedim(0, -1).cpu()

# Slicing for display if needed
if run_in_3d:
    ct = ct[..., slice_index]
    labels = labels[..., slice_index]
    labels_smoothed = labels_smoothed[..., slice_index]

# Display slices
plt.subplot(131).axis("off")
plt.title("input image")
plt.imshow(ct)
plt.subplot(132).axis("off")
plt.title("input labels")
plt.imshow(labels)
plt.subplot(133).axis("off")
plt.title("output labels")
plt.imshow(labels_smoothed)
plt.show()
