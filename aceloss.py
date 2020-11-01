import time

import numpy as np
import torch
import torch.nn as nn


class ACLoss(nn.Module):
    """
    Active Contour Loss
    based on sobel filter
    """

    def __init__(self, miu=1.0, classes=3):
        super(ACLoss, self).__init__()

        self.miu = miu
        self.classes = classes
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        self.sobel_x = nn.Parameter(torch.from_numpy(sobel_x).float().expand(self.classes, 1, 3, 3),
                                    requires_grad=False)
        self.sobel_y = nn.Parameter(torch.from_numpy(sobel_y).float().expand(self.classes, 1, 3, 3),
                                    requires_grad=False)

        self.diff_x = nn.Conv2d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.diff_x.weight = self.sobel_x
        self.diff_y = nn.Conv2d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.diff_y.weight = self.sobel_y

    def forward(self, predication, label):
        grd_x = self.diff_x(predication)
        grd_y = self.diff_y(predication)

        # length
        length = torch.sum(
            torch.abs(torch.sqrt(grd_x ** 2 + grd_y ** 2 + 1e-8)))
        length = (length - length.min()) / (length.max() - length.min() + 1e-8)
        length = torch.sum(length)

        # region
        label = label.float()
        c_in = torch.ones_like(predication)
        c_out = torch.zeros_like(predication)
        region_in = torch.abs(torch.sum(predication * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - predication) * ((label - c_out) ** 2)))
        region = self.miu * region_in + region_out

        return region + length


class ACLossV2(nn.Module):
    """
    Active Contour Loss
    based on maxpooling & minpooling
    """

    def __init__(self, miu=1.0, classes=3):
        super(ACLossV2, self).__init__()

        self.miu = miu
        self.classes = classes

    def forward(self, predication, label):
        min_pool_x = nn.functional.max_pool2d(
            predication * -1, (3, 3), 1, 1) * -1
        contour = torch.relu(nn.functional.max_pool2d(
            min_pool_x, (3, 3), 1, 1) - min_pool_x)

        # length
        length = torch.sum(torch.abs(contour))

        # region
        label = label.float()
        c_in = torch.ones_like(predication)
        c_out = torch.zeros_like(predication)
        region_in = torch.abs(torch.sum(predication * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - predication) * ((label - c_out) ** 2)))
        region = self.miu * region_in + region_out

        return region + length


def ACELoss(y_pred, y_true, u=1, a=1, b=1):
    """
    Active Contour Loss
    based on total variations and mean curvature
    """
    def first_derivative(input):
        u = input
        m = u.shape[2]
        n = u.shape[3]

        ci_0 = (u[:, :, 1, :] - u[:, :, 0, :]).unsqueeze(2)
        ci_1 = u[:, :, 2:, :] - u[:, :, 0:m - 2, :]
        ci_2 = (u[:, :, -1, :] - u[:, :, m - 2, :]).unsqueeze(2)
        ci = torch.cat([ci_0, ci_1, ci_2], 2) / 2

        cj_0 = (u[:, :, :, 1] - u[:, :, :, 0]).unsqueeze(3)
        cj_1 = u[:, :, :, 2:] - u[:, :, :, 0:n - 2]
        cj_2 = (u[:, :, :, -1] - u[:, :, :, n - 2]).unsqueeze(3)
        cj = torch.cat([cj_0, cj_1, cj_2], 3) / 2

        return ci, cj

    def second_derivative(input, ci, cj):
        u = input
        m = u.shape[2]
        n = u.shape[3]

        cii_0 = (u[:, :, 1, :] + u[:, :, 0, :] -
                 2 * u[:, :, 0, :]).unsqueeze(2)
        cii_1 = u[:, :, 2:, :] + u[:, :, :-2, :] - 2 * u[:, :, 1:-1, :]
        cii_2 = (u[:, :, -1, :] + u[:, :, -2, :] -
                 2 * u[:, :, -1, :]).unsqueeze(2)
        cii = torch.cat([cii_0, cii_1, cii_2], 2)

        cjj_0 = (u[:, :, :, 1] + u[:, :, :, 0] -
                 2 * u[:, :, :, 0]).unsqueeze(3)
        cjj_1 = u[:, :, :, 2:] + u[:, :, :, :-2] - 2 * u[:, :, :, 1:-1]
        cjj_2 = (u[:, :, :, -1] + u[:, :, :, -2] -
                 2 * u[:, :, :, -1]).unsqueeze(3)

        cjj = torch.cat([cjj_0, cjj_1, cjj_2], 3)

        cij_0 = ci[:, :, :, 1:n]
        cij_1 = ci[:, :, :, -1].unsqueeze(3)

        cij_a = torch.cat([cij_0, cij_1], 3)
        cij_2 = ci[:, :, :, 0].unsqueeze(3)
        cij_3 = ci[:, :, :, 0:n - 1]
        cij_b = torch.cat([cij_2, cij_3], 3)
        cij = cij_a - cij_b

        return cii, cjj, cij

    def region(y_pred, y_true, u=1):
        label = y_true.float()
        c_in = torch.ones_like(y_pred)
        c_out = torch.zeros_like(y_pred)
        region_in = torch.abs(torch.sum(y_pred * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - y_pred) * ((label - c_out) ** 2)))
        region = u * region_in + region_out
        return region

    def elastica(input, a=1, b=1):
        ci, cj = first_derivative(input)
        cii, cjj, cij = second_derivative(input, ci, cj)
        beta = 1e-8
        length = torch.sqrt(beta + ci ** 2 + cj ** 2)
        curvature = (beta + ci ** 2) * cjj + \
                    (beta + cj ** 2) * cii - 2 * ci * cj * cij
        curvature = torch.abs(curvature) / ((ci ** 2 + cj ** 2) ** 1.5 + beta)
        elastica = torch.sum((a + b * (curvature ** 2)) * torch.abs(length))
        return elastica

    loss = region(y_pred, y_true, u=u) + elastica(y_pred, a=a, b=b)
    return loss


class ACLoss3D(nn.Module):
    """
    Active Contour Loss
    based on sobel filter
    """

    def __init__(self, classes=4, alpha=1):
        super(ACLoss3D, self).__init__()
        self.alpha = alpha
        self.classes = classes
        sobel = np.array([[[1., 2., 1.],
                           [2., 4., 2.],
                           [1., 2., 1.]],

                          [[0., 0., 0.],
                           [0., 0., 0.],
                           [0., 0., 0.]],

                          [[-1., -2., -1.],
                           [-2., -4., -2.],
                           [-1., -2., -1.]]])

        self.sobel_x = nn.Parameter(
            torch.from_numpy(sobel.transpose(0, 1, 2)).float().unsqueeze(0).unsqueeze(0).expand(self.classes, 1, 3, 3,
                                                                                                3), requires_grad=False)
        self.sobel_y = nn.Parameter(
            torch.from_numpy(sobel.transpose(1, 0, 2)).float().unsqueeze(0).unsqueeze(0).expand(self.classes, 1, 3, 3,
                                                                                                3), requires_grad=False)
        self.sobel_z = nn.Parameter(
            torch.from_numpy(sobel.transpose(1, 2, 0)).float().unsqueeze(0).unsqueeze(0).expand(self.classes, 1, 3, 3,
                                                                                                3), requires_grad=False)

        self.diff_x = nn.Conv3d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.diff_x.weight = self.sobel_x
        self.diff_y = nn.Conv3d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.diff_y.weight = self.sobel_y
        self.diff_z = nn.Conv3d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.diff_z.weight = self.sobel_z

    def forward(self, predication, label):
        grd_x = self.diff_x(predication)
        grd_y = self.diff_y(predication)
        grd_z = self.diff_z(predication)

        # length
        length = torch.sqrt(grd_x ** 2 + grd_y ** 2 + grd_z ** 2 + 1e-8)
        length = (length - length.min()) / (length.max() - length.min() + 1e-8)
        length = torch.sum(length)

        # region
        label = label.float()
        c_in = torch.ones_like(predication)
        c_out = torch.zeros_like(predication)
        region_in = torch.abs(torch.sum(predication * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - predication) * ((label - c_out) ** 2)))
        region = region_in + region_out

        return self.alpha * region + length


class ACLoss3DV2(nn.Module):
    """
    Active Contour Loss
    based on minpooling & maxpooling
    """

    def __init__(self, miu=1.0, classes=3):
        super(ACLoss3DV2, self).__init__()

        self.miu = miu
        self.classes = classes

    def forward(self, predication, label):
        min_pool_x = nn.functional.max_pool3d(
            predication * -1, (3, 3, 3), 1, 1) * -1
        contour = torch.relu(nn.functional.max_pool3d(
            min_pool_x, (3, 3, 3), 1, 1) - min_pool_x)

        # length
        length = torch.sum(torch.abs(contour))

        # region
        label = label.float()
        c_in = torch.ones_like(predication)
        c_out = torch.zeros_like(predication)
        region_in = torch.abs(torch.sum(predication * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - predication) * ((label - c_out) ** 2)))
        region = self.miu * region_in + region_out

        return region + length


class ACELoss3D(nn.Module):
    """
    Active contour based elastic model loss
    based on total variations and mean curvature
    """

    def __init__(self, alpha=1e-3, beta=1.0, miu=1, classes=3):
        super(ACELoss3D, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.miu = miu

    def first_derivative(self, input):
        u = input
        m = u.shape[2]
        n = u.shape[3]
        k = u.shape[4]

        ci_0 = (u[:, :, 1, :, :] - u[:, :, 0, :, :]).unsqueeze(2)
        ci_1 = u[:, :, 2:, :, :] - u[:, :, 0:m - 2, :, :]
        ci_2 = (u[:, :, -1, :, :] - u[:, :, m - 2, :, :]).unsqueeze(2)
        ci = torch.cat([ci_0, ci_1, ci_2], 2) / 2

        cj_0 = (u[:, :, :, 1, :] - u[:, :, :, 0, :]).unsqueeze(3)
        cj_1 = u[:, :, :, 2:, :] - u[:, :, :, 0:n - 2, :]
        cj_2 = (u[:, :, :, -1, :] - u[:, :, :, n - 2, :]).unsqueeze(3)
        cj = torch.cat([cj_0, cj_1, cj_2], 3) / 2

        ck_0 = (u[:, :, :, :, 1] - u[:, :, :, :, 0]).unsqueeze(4)
        ck_1 = u[:, :, :, :, 2:] - u[:, :, :, :, 0:k - 2]
        ck_2 = (u[:, :, :, :, -1] - u[:, :, :, :, k - 2]).unsqueeze(4)
        ck = torch.cat([ck_0, ck_1, ck_2], 4) / 2

        return ci, cj, ck

    def second_derivative(self, input, ci, cj, ck):
        u = input
        m = u.shape[2]
        n = u.shape[3]
        k = u.shape[4]

        cii_0 = (u[:, :, 1, :, :] + u[:, :, 0, :, :] -
                 2 * u[:, :, 0, :, :]).unsqueeze(2)
        cii_1 = u[:, :, 2:, :, :] + \
            u[:, :, :-2, :, :] - 2 * u[:, :, 1:-1, :, :]
        cii_2 = (u[:, :, -1, :, :] + u[:, :, -2, :, :] -
                 2 * u[:, :, -1, :, :]).unsqueeze(2)
        cii = torch.cat([cii_0, cii_1, cii_2], 2)

        cjj_0 = (u[:, :, :, 1, :] + u[:, :, :, 0, :] -
                 2 * u[:, :, :, 0, :]).unsqueeze(3)
        cjj_1 = u[:, :, :, 2:, :] + \
            u[:, :, :, :-2, :] - 2 * u[:, :, :, 1:-1, :]
        cjj_2 = (u[:, :, :, -1, :] + u[:, :, :, -2, :] -
                 2 * u[:, :, :, -1, :]).unsqueeze(3)

        cjj = torch.cat([cjj_0, cjj_1, cjj_2], 3)

        ckk_0 = (u[:, :, :, :, 1] + u[:, :, :, :, 0] -
                 2 * u[:, :, :, :, 0]).unsqueeze(4)
        ckk_1 = u[:, :, :, :, 2:] + \
            u[:, :, :, :, :-2] - 2 * u[:, :, :, :, 1:-1]
        ckk_2 = (u[:, :, :, :, -1] + u[:, :, :, :, -2] -
                 2 * u[:, :, :, :, -1]).unsqueeze(4)

        ckk = torch.cat([ckk_0, ckk_1, ckk_2], 4)

        cij_0 = ci[:, :, :, 1:n, :]
        cij_1 = ci[:, :, :, -1, :].unsqueeze(3)

        cij_a = torch.cat([cij_0, cij_1], 3)
        cij_2 = ci[:, :, :, 0, :].unsqueeze(3)
        cij_3 = ci[:, :, :, 0:n - 1, :]
        cij_b = torch.cat([cij_2, cij_3], 3)
        cij = cij_a - cij_b

        cik_0 = ci[:, :, :, :, 1:n]
        cik_1 = ci[:, :, :, :, -1].unsqueeze(4)

        cik_a = torch.cat([cik_0, cik_1], 4)
        cik_2 = ci[:, :, :, :, 0].unsqueeze(4)
        cik_3 = ci[:, :, :, :, 0:k - 1]
        cik_b = torch.cat([cik_2, cik_3], 4)
        cik = cik_a - cik_b

        cjk_0 = cj[:, :, :, :, 1:n]
        cjk_1 = cj[:, :, :, :, -1].unsqueeze(4)

        cjk_a = torch.cat([cjk_0, cjk_1], 4)
        cjk_2 = cj[:, :, :, :, 0].unsqueeze(4)
        cjk_3 = cj[:, :, :, :, 0:k - 1]
        cjk_b = torch.cat([cjk_2, cjk_3], 4)
        cjk = cjk_a - cjk_b

        return cii, cjj, ckk, cij, cik, cjk

    def region(self, y_pred, y_true, u=1):
        label = y_true.float()
        c_in = torch.ones_like(y_pred)
        c_out = torch.zeros_like(y_pred)
        region_in = torch.abs(torch.sum(y_pred * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - y_pred) * ((label - c_out) ** 2)))
        region = u * region_in + region_out
        return region

    def elastica(self, input, a=1, b=1):
        ci, cj, ck = self.first_derivative(input)
        cii, cjj, ckk, cij, cik, cjk = self.second_derivative(
            input, ci, cj, ck)
        beta = 1e-8
        length = torch.sqrt(beta + ci ** 2 + cj ** 2 + ck ** 2)
        curvature = (1 + ci ** 2 + cj ** 2) * ckk + (1 + cj ** 2 + ck ** 2) * cii + (
            1 + ci ** 2 + ck ** 2) * cjj - 2 * cik * cjk * cij
        curvature = torch.abs(curvature) / \
            ((1 + ci ** 2 + cj ** 2 + ck ** 2) ** 0.5 + beta)
        elastica = torch.sum(a + b * (curvature ** 2) * torch.abs(length))
        return elastica

    def forward(self, y_pred, y_true):
        loss = self.region(y_pred, y_true, u=self.miu) + \
            self.elastica(y_pred, a=self.alpha, b=self.beta)
        return loss


class FastACELoss3D(nn.Module):
    """
    Active contour based elastic model loss
    based on sobel and laplace filter
    """

    def __init__(self, miu=1, alpha=1e-3, beta=2.0, classes=4, types="laplace"):
        super(FastACELoss3D, self).__init__()
        self.miu = miu
        self.alpha = alpha
        self.beta = beta
        self.classes = classes
        self.types = types
        sobel = np.array([[[1., 2., 1.],
                           [2., 4., 2.],
                           [1., 2., 1.]],

                          [[0., 0., 0.],
                           [0., 0., 0.],
                           [0., 0., 0.]],

                          [[-1., -2., -1.],
                           [-2., -4., -2.],
                           [-1., -2., -1.]]])
        laplace_kernel = np.ones((3, 3, 3))
        laplace_kernel[1, 1, 1] = -26

        self.sobel_x = nn.Parameter(
            torch.from_numpy(sobel.transpose(0, 1, 2)).float().unsqueeze(0).unsqueeze(0).expand(self.classes, 1, 3, 3,
                                                                                                3), requires_grad=False)
        self.sobel_y = nn.Parameter(
            torch.from_numpy(sobel.transpose(1, 0, 2)).float().unsqueeze(0).unsqueeze(0).expand(self.classes, 1, 3, 3,
                                                                                                3), requires_grad=False)
        self.sobel_z = nn.Parameter(
            torch.from_numpy(sobel.transpose(1, 2, 0)).float().unsqueeze(0).unsqueeze(0).expand(self.classes, 1, 3, 3,
                                                                                                3), requires_grad=False)
        self.laplace = nn.Parameter(
            torch.from_numpy(laplace_kernel).float().unsqueeze(
                0).unsqueeze(0).expand(self.classes, 1, 3, 3, 3),
            requires_grad=False)

        self.diff_x = nn.Conv3d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.diff_x.weight = self.sobel_x
        self.diff_y = nn.Conv3d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.diff_y.weight = self.sobel_y
        self.diff_z = nn.Conv3d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.diff_z.weight = self.sobel_z

        self.laplace_operator = nn.Conv3d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1,
                                          padding=1,
                                          bias=False)
        self.laplace_operator.weight = self.laplace

    def forward(self, predication, label):
        grd_x = self.diff_x(predication)
        grd_y = self.diff_y(predication)
        grd_z = self.diff_z(predication)
        diff = self.laplace_operator(predication)

        # length
        length = torch.sqrt(grd_x ** 2 + grd_y ** 2 + grd_z ** 2 + 1e-8)
        length = (length - length.min()) / (length.max() - length.min() + 1e-8)

        # curvature
        if self.types:
            curvature = torch.abs(diff)
            curvature = (curvature - curvature.min()) / \
                (curvature.max() - curvature.min() + 1e-8)
        else:
            """
            maybe more powerful
            """
            curvature = torch.abs(
                diff) / ((grd_x ** 2 + grd_y ** 2 + grd_z ** 2 + 1) ** 0.5 + 1e-8)
            curvature = (curvature - curvature.min()) / \
                (curvature.max() - curvature.min() + 1e-8)
        # region
        label = label.float()
        c_in = torch.ones_like(predication)
        c_out = torch.zeros_like(predication)
        region_in = torch.abs(torch.sum(predication * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - predication) * ((label - c_out) ** 2)))
        region = self.miu * region_in + region_out

        # elastic
        elastic = torch.sum((self.alpha + self.beta * curvature ** 2) * length)
        return region + elastic


class FastACELoss3DV2(nn.Module):
    """
    Active contour based elastic model loss
    based on minpooling & maxpooling and laplace filter
    """

    def __init__(self, miu=1, alpha=1e-3, beta=2.0, classes=4, types="other"):
        super(FastACELoss3DV2, self).__init__()
        self.miu = miu
        self.alpha = alpha
        self.beta = beta
        self.classes = classes
        self.types = types
        laplace_kernel = np.ones((3, 3, 3))
        laplace_kernel[1, 1, 1] = -26

        self.laplace = nn.Parameter(
            torch.from_numpy(laplace_kernel).float().unsqueeze(
                0).unsqueeze(0).expand(self.classes, 1, 3, 3, 3),
            requires_grad=False)

        self.laplace_operator = nn.Conv3d(self.classes, self.classes, groups=self.classes, kernel_size=3, stride=1,
                                          padding=1,
                                          bias=False)
        self.laplace_operator.weight = self.laplace

    def forward(self, predication, label):
        min_pool_x = nn.functional.max_pool3d(predication * -1, 3, 1, 1) * -1
        contour = torch.relu(nn.functional.max_pool3d(
            min_pool_x, 3, 1, 1) - min_pool_x)

        diff = self.laplace_operator(predication)

        # length
        length = torch.abs(contour)

        # curvature
        if self.types:
            curvature = torch.abs(diff)
            curvature = (curvature - curvature.min()) / \
                (curvature.max() - curvature.min() + 1e-8)
        else:
            """
            maybe more powerful
            """
            curvature = torch.abs(diff) / ((length ** 2 + 1) ** 0.5 + 1e-8)
            curvature = (curvature - curvature.min()) / \
                (curvature.max() - curvature.min() + 1e-8)
        # region
        label = label.float()
        c_in = torch.ones_like(predication)
        c_out = torch.zeros_like(predication)
        region_in = torch.abs(torch.sum(predication * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - predication) * ((label - c_out) ** 2)))
        region = self.miu * region_in + region_out

        # elastic
        elastic = torch.sum((self.alpha + self.beta * curvature ** 2) * length)
        return region + elastic


"test demo"
x2 = torch.rand((2, 3, 97, 80))
y2 = torch.rand((2, 3, 97, 80))
time1 = time.time()
print("ACLoss:", ACLoss()(x2, y2).item())
print(time.time() - time1)
time2 = time.time()
print("ACLossV2:", ACLossV2()(x2, y2).item())
print(time.time() - time2)
time3 = time.time()
print("ACELoss:", ACELoss(x2, y2).item())
print(time.time() - time3)
time6 = time.time()
x3 = torch.rand((2, 4, 112, 97, 80))
y3 = torch.rand((2, 4, 112, 97, 80))
time7 = time.time()
print("ACLoss3D:", ACLoss3D()(x3, y3).item())
print(time.time() - time7)
time8 = time.time()
print("ACLoss3DV2:", ACLoss3DV2()(x3, y3).item())
print(time.time() - time8)
time9 = time.time()
print("ACELoss3D:", ACELoss3D().cuda()(x3.cuda(), y3.cuda()).item())
print(time.time() - time9)
time10 = time.time()
print("FastACELoss3D:", FastACELoss3D().cuda()(x3.cuda(), y3.cuda()).item())
print(time.time() - time10)
time11 = time.time()
print("FastACELoss3DV2:", FastACELoss3DV2().cuda()(x3.cuda(), y3.cuda()).item())
print(time.time() - time11)
time12 = time.time()
