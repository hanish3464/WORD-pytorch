import cv2
import numpy as np
import config

def gaussian():

  sigma = 10
  spread = config.gaussian_spread

  extent = int(spread * sigma)
  isotropicGaussian2dMap = np.zeros((2*extent,2*extent), dtype=np.float32)

  for i in range(2 * extent):
    for j in range(2 * extent):
      isotropicGaussian2dMap[i, j] = float(1) / 2 / np.pi / (sigma ** 2) * np.exp(
        -1 / 2 * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2) / (sigma ** 2))
  isotropicGaussian2dMap = (isotropicGaussian2dMap / np.max(isotropicGaussian2dMap) * 255).astype(np.uint8)

  #repair cropped pixel value
  h, w = isotropicGaussian2dMap.shape
  adjust_gaussian_heat_map = np.zeros((h+2,w+2)).astype(np.uint8)
  adjust_gaussian_heat_map[:h, :w] = isotropicGaussian2dMap[:,:]

  adjust_gaussian_heat_map[:h,w] = isotropicGaussian2dMap[:,1]
  adjust_gaussian_heat_map[:h,w+1] = isotropicGaussian2dMap[:, 0]
  adjust_gaussian_heat_map[h+1] = adjust_gaussian_heat_map[0]
  adjust_gaussian_heat_map[h] = adjust_gaussian_heat_map[1]


  # Convert Grayscale to HeatMap Using Opencv
  isotropicGaussianHeatmap = cv2.applyColorMap(adjust_gaussian_heat_map, cv2.COLORMAP_JET)
  cv2.imwrite('./gaussian_heapmap.jpg', isotropicGaussianHeatmap)

  return adjust_gaussian_heat_map

gaussian()