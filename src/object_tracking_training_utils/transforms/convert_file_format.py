import numpy as np
import torch


class ConvertFileFormat:

    def __init__(self):
        print(" I am inside init function of test file ")

    @staticmethod
    def to_numpy(tensor):
        """
        Converts a PyTorch tensor to a NumPy array.

        Args:
            tensor (torch.Tensor or numpy.ndarray): The tensor to be converted.
        Returns:
            numpy.ndarray: The converted NumPy array if the input is a PyTorch tensor.
                        If the input is already a NumPy array, it is returned unchanged.

        Raises:
            ValueError: If the input is neither a PyTorch tensor nor a NumPy array.
        """
        
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        elif type(tensor).__module__ != 'numpy':
            raise ValueError("Cannot convert {} to numpy array"
                             .format(type(tensor)))
        return tensor

    @staticmethod
    def torch_var_to_numpy(tensor_var):
        """ Converts a Pytorch tensor to a NumPy array with dimensions rearrangement.

        Args:
            tensor_var (torch.Tensor): A Pytorch tensor. Normally a batch of 3D (Ch x Height x Width) images in the format (Batch, Channels, Height, Width).
        Returns:
            numpy.ndarray: The tensor converted to a NumPy array with dimensions rearranged to (Batch, Height, Width, Channels).
        """
        if torch.is_tensor(tensor_var):
            np_tensor = tensor_var.detach().cpu().numpy()
            np_tensor = np.transpose(np_tensor, (0, 2, 3, 1))
            np_tensor = np_tensor.squeeze()
            return np_tensor
        elif type(tensor_var).__module__ != 'numpy':
            raise ValueError("Cannot convert {} to numpy array"
                             .format(type(tensor_var)))
        return tensor_var

    @staticmethod
    def numpy_to_torch_var(np_tensor, device):
        """ Converts a NumPy array to a PyTorch tensor with appropriate dimension rearrangement.

        Args:
            np_tensor (numpy.ndarray): The NumPy array representing the tensor
            device (torch.device): The device where the tensor will be stored.
        Returns:
            torch.Tensor: The equivalent Pytorch tensor with dimensions formatted as 
                          (Batch, Channels, Height, Width).
        """
        if len(np_tensor.shape) == 3:
            np_tensor = np.expand_dims(np_tensor, axis=0)
        var = np.transpose(np_tensor, (0, 3, 1, 2))
        var = torch.from_numpy(var).float()
        var = var.to(device)
        return var

    @staticmethod
    def permute_tensor_dims(input_var):
        """ Rearranges the dimensions of a NumPy array for PyTorch compatibility.
            H x W x C -> C x H x W.

        Args:
            input_var (numpy.ndarray): The NumPy array in the format (H, W, C).
        Returns:
            torch.Tensor: A Pytorch tensor in the format (C, H, W).
        """
        var = torch.FloatTensor(np.array(input_var)).permute(2, 0, 1)
        return var

    @staticmethod
    def to_torch(ndarray):
        """
        Converts a NumPy array to a PyTorch tensor if it is not already a tensor.

        Args:
            ndarray (numpy.ndarray or torch.Tensor): The input data to be converted to a PyTorch tensor.
        Returns:
            torch.Tensor: A PyTorch tensor. If the input is already a tensor, it is returned as is.
        Raises:
            ValueError: If the input is neither a NumPy array nor a PyTorch tensor.

        """
        if type(ndarray).__module__ == 'numpy':
            return torch.from_numpy(ndarray)
        elif not torch.is_tensor(ndarray):
            raise ValueError("Cannot convert {} to torch tensor"
                             .format(type(ndarray)))
        return ndarray

    @staticmethod
    def im_to_numpy(img):
        """
        Converts an image from PyTorch tensor to a NumPy array in the standard image format.

        Args:
            img (torch.Tensor or numpy.ndarray): The image tensor or array. 
                                                If it is a tensor, it must be in the format (C, H, W).
        Returns:
            numpy.ndarray: The image as a NumPy array in the format (H, W, C).
        """
        img = ConvertFileFormat.to_numpy(img)
        img = np.transpose(img, (1, 2, 0))  # H*W*C
        return img

    @staticmethod
    def im_to_torch(img):
        """
        Converts an image from OpenCV/Numpy format (H, W, C) to PyTorch format (C, H, W).

        Args:
            img (numpy.ndarray): The image as a NumPy array with the format (H, W, C).
        Returns:
            torch.Tensor: The image as a PyTorch tensor with the format (C, H, W).
        """
        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = ConvertFileFormat.to_torch(img).float()
        return img

    @staticmethod
    def torch_to_img(img):
        """
        Converts a PyTorch tensor to a NumPy array in image format.

        Args:
            img (torch.Tensor): The image tensor, typically in the format (C, H, W).
        Returns:
            numpy.ndarray: The image as a NumPy array in the format (H, W, C).
        """
        img = ConvertFileFormat.to_numpy(torch.squeeze(img, 0))
        img = np.transpose(img, (1, 2, 0))  # H*W*C
        return img
