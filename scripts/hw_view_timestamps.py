import numpy as np
import matplotlib.pyplot as plt

def main():
  dir = "../data/hw/in/"

  imu_data = np.loadtxt(dir + "hw_imu.txt", dtype=np.float64)
  apriltag_data = np.loadtxt(dir + "hw_apriltag.txt", dtype=np.float64)
  mocap_data = np.loadtxt(dir + "hw_rover_truth.txt", dtype=np.float64)

  plt.figure()
  plt.scatter(imu_data[:,0], np.zeros(imu_data.shape[0]), marker='|')
  plt.title("IMU timestamps")

  plt.figure()
  plt.scatter(apriltag_data[:,1], apriltag_data[:,0], marker='|', c=apriltag_data[:,0])
  plt.scatter(apriltag_data[:,1], -np.ones(apriltag_data.shape[0]), marker='|', c=apriltag_data[:,0])
  plt.title("Apriltag timestamps")

  plt.figure()
  plt.scatter(mocap_data[:,0], np.zeros(mocap_data.shape[0]), marker='|')
  plt.title("Mocap timestamps")

  plt.show()


if __name__=="__main__":
  main()