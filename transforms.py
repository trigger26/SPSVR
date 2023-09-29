import numbers
import math
import torch
import torchvision.transforms.functional as F


class GroupRandomResizedCrop(torch.nn.Module):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        _, height, width = img.size()
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(images[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size) for img in images]


class GroupRandomRotation(torch.nn.Module):
    def __init__(
            self, degrees, expand=False, center=None, fill=0, resample=None
    ):
        super().__init__()

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        self.center = center

        self.resample = resample
        self.expand = expand

        self.fill = fill

    @staticmethod
    def get_params(degrees) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        angle = self.get_params(self.degrees)

        return [F.rotate(img, angle=angle, resample=self.resample, expand=self.expand, center=self.center, fill=fill) for img in images]


class GroupRandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return [F.hflip(img) for img in images]
        return images


class GroupRandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return [F.vflip(img) for img in images]
        return images


class GroupColorJitter(torch.nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                images = [F.adjust_brightness(img, brightness_factor) for img in images]
            elif fn_id == 1 and contrast_factor is not None:
                images = [F.adjust_contrast(img, contrast_factor) for img in images]
            elif fn_id == 2 and saturation_factor is not None:
                images = [F.adjust_saturation(img, saturation_factor) for img in images]
            elif fn_id == 3 and hue_factor is not None:
                images = [F.adjust_hue(img, hue_factor) for img in images]

        return images


class GroupGaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min, sigma_max):
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return [F.gaussian_blur(img, self.kernel_size, [sigma, sigma]) for img in images]


class GroupToTensor(torch.nn.Module):
    def __call__(self, images):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return [F.to_tensor(img) for img in images]


class GroupNormalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return torch.stack([F.normalize(tensor, self.mean, self.std, self.inplace) for tensor in tensors])


class GroupRandomErasing(torch.nn.Module):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=None):
        img_c, img_h, img_w = img.size()
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def forward(self, images):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value, ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            x, y, h, w, v = self.get_params(images[0], scale=self.scale, ratio=self.ratio, value=value)
            if x == 0 and y == 0:
                return images
            else:
                return [F.erase(img, x, y, h, w, v, self.inplace) for img in images]
        return images


class GroupCenterCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return [F.center_crop(img, self.size) for img in images]


class GroupResize(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        return [F.resize(img, self.size) for img in images]


class GroupConvertImageDtype(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def __call__(self, images):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return [F.convert_image_dtype(img, self.dtype) for img in images]


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]

    return [float(d) for d in x]
