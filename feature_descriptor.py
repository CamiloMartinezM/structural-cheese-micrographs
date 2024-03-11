# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 22:20:26 2021

@author: Camilo Martínez
"""
import scipy.ndimage
import numpy as np


def multiscale_statistics(img: np.ndarray, scales: int) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Input image.
        scales (int, optional): Number of scales to consider, i.e neighboring window radius 
                                in pixels.
                                
    Returns:
        np.ndarray: Feature vectors of the input image of shape (*img.shape, 4*3*scales)
    """
    directions = 4
    gx_img, gy_img = np.gradient(img)

    feature_vectors = np.zeros((img.size * 3 * directions * scales,), dtype=np.float64)
    computed_statistics_per_scale = np.zeros(
        (scales, img.shape[0], img.shape[1], directions * 3), dtype=np.float64
    )
    for scale in range(1, scales + 1):
        computed_statistics_per_dir = np.zeros(
            (directions, *img.shape, 3), dtype=np.float64
        )
        filter_size = 2 * (scale - 1) + 3
        orig = np.zeros((filter_size, filter_size), dtype=np.float64)
        orig[:, 0] = 1 / filter_size
        for c in range(directions):
            # c = 0 -> North; c = 1 -> East; c = 2 -> South; c = 3 -> West
            correlation_filter = np.rot90(orig, 3 - c, (0, 1))
            convolution_filter = np.flip(correlation_filter)

            directions_img = scipy.ndimage.convolve(
                img, convolution_filter, cval=0.0, mode="constant"
            )
            directions_gx_img = scipy.ndimage.convolve(
                gx_img, convolution_filter, cval=0.0, mode="constant"
            )
            directions_gy_img = scipy.ndimage.convolve(
                gy_img, convolution_filter, cval=0.0, mode="constant"
            )

            computed_statistics_per_dir[c] = np.concatenate(
                (
                    directions_img[..., np.newaxis],
                    directions_gx_img[..., np.newaxis],
                    directions_gy_img[..., np.newaxis],
                ),
                axis=-1,
            )

        computed_statistics_per_scale[scale - 1] = np.concatenate(
            [
                computed_statistics_per_dir[i][..., np.newaxis]
                for i in range(directions)
            ],
            axis=-1,
        ).reshape(*img.shape, 3 * directions)

    for i in range(scales):
        feature_vectors[i::scales] = computed_statistics_per_scale[i].ravel()
    
    return feature_vectors.reshape((*img.shape, 3 * directions * scales))

def naive_approach(img: np.ndarray, scales: int) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Input image.
        scales (int, optional): Number of scales to consider, i.e neighboring window radius 
                                in pixels.
                                
    Returns:
        np.ndarray: Feature vectors of the input image of shape (*img.shape, 4*3*scales)
    """
    padded_img = np.pad(img, scales, mode="constant")
    gx, gy = np.gradient(padded_img)
    feature_vectors = np.zeros((img.shape[0] * img.shape[1], 4 * 3 * scales))
    z = 0
    for i in range(scales, padded_img.shape[0] - scales):
        for j in range(scales, padded_img.shape[1] - scales):
            for scale in range(1, scales + 1):
                N = padded_img[i - scale, j - scale : j + scale + 1]
                E = padded_img[i - scale : i + scale + 1, j + scale]
                S = padded_img[i + scale, j - scale : j + scale + 1]
                W = padded_img[i - scale : i + scale + 1, j - scale]

                N_gx = gx[i - scale, j - scale : j + scale + 1]
                E_gx = gx[i - scale : i + scale + 1, j + scale]
                S_gx = gx[i + scale, j - scale : j + scale + 1]
                W_gx = gx[i - scale : i + scale + 1, j - scale]

                N_gy = gy[i - scale, j - scale : j + scale + 1]
                E_gy = gy[i - scale : i + scale + 1, j + scale]
                S_gy = gy[i + scale, j - scale : j + scale + 1]
                W_gy = gy[i - scale : i + scale + 1, j - scale]

                neighbors = np.vstack((N, E, S, W))
                avgs = np.mean(neighbors, axis=1)

                neighbors_gx = np.vstack((N_gx, E_gx, S_gx, W_gx))
                grads_x = np.mean(neighbors_gx, axis=1)

                neighbors_gy = np.vstack((N_gy, E_gy, S_gy, W_gy))
                grads_y = np.mean(neighbors_gy, axis=1)

                feature_vectors[z, 4 * 3 * (scale - 1) : 4 * 3 * scale] = np.ravel(
                    (avgs, grads_x, grads_y), "F"
                )
            z += 1
            
    return feature_vectors.reshape((*img.shape, 4 * 3 * scales))