import argparse
import os
import numpy as np
import rosbag

def main(bagfile, start, end, outfile, imu_ds_factor, apriltag_ds_factor):
  bag = rosbag.Bag(bagfile)
  topics = ['/april0_ned', '/april1_ned', '/april2_ned', '/april3_ned', '/realsense_ned', '/tag_detections', '/camera/imu']
  msgs = {key: [] for key in topics}

  for topic, msg, t in bag.read_messages(topics=topics):
    msgs[topic].append(msg)

  bag.close()

  first_times = []
  for topic, msg_list in msgs.items():
    first_times.append(msg_list[0].header.stamp.secs + 1e-9 * msg_list[0].header.stamp.nsecs)
  init_time = min(first_times)

  assert outfile.split('.')[-1] == 'txt', f"Provided path must end with a .txt file, got: {outfile}"
  outpath = os.path.join(*outfile.split('.')[:-1])

  apriltag_period = 1.0/30.0
  with open(outpath + "_apriltag.txt", 'w') as f:
    num_skipped = apriltag_ds_factor
    last_time_used = -np.inf
    for msg in msgs[topics[5]]:
      if len(msg.detections) == 0:
        continue

      stamp = msg.header.stamp
      time = stamp.secs + 1e-9 * stamp.nsecs - init_time
      if time < start or time > end:
        continue
      if num_skipped < apriltag_ds_factor - 1 and not (time - last_time_used) > (apriltag_period * apriltag_ds_factor - 5e-3):
        num_skipped += 1
        continue
      num_skipped = 0
      last_time_used = time
      
      for tag in msg.detections:
        position = tag.pose.pose.pose.position
        orientation = tag.pose.pose.pose.orientation

        f.write("{} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
          tag.id[0], time, position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z))
        f.write("\n")
  
  imu_period = 1.0/200.0
  with open(outpath + "_imu.txt", 'w') as f:
    num_skipped = imu_ds_factor
    last_time_used = -np.inf
    for msg in msgs[topics[6]]:
      stamp = msg.header.stamp
      time = stamp.secs + 1e-9 * stamp.nsecs - init_time
      if time < start or time > end:
        continue
      if num_skipped < imu_ds_factor - 1 and not (time - last_time_used) > (imu_period * imu_ds_factor - 1e-3):
        num_skipped += 1
        continue
      num_skipped = 0
      last_time_used = time

      f.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
        time, msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z, msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))
      f.write("\n")

  for i in range(4):
    with open(outpath + "_april" + str(i) + ".txt", 'w') as f:
      for msg in msgs[topics[i]]:
        stamp = msg.header.stamp
        time = stamp.secs + 1e-9 * stamp.nsecs - init_time
        if time < start or time > end:
          continue

        f.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
          time, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z))
        f.write("\n")

  with open(outpath + "_rover_truth.txt", 'w') as f:
    for msg in msgs[topics[4]]:
      stamp = msg.header.stamp
      time = stamp.secs + 1e-9 * stamp.nsecs - init_time - 0.94 # there is a significant time offset between mocap and cam, not sure why
      if time < start or time > end:
        continue

      f.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
        time, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z))
      f.write("\n")

  print(f"Data written to {outpath}")

  

if __name__ == "__main__":
    # Parse through arguments
    parser = argparse.ArgumentParser(description="Convert a ROS1 rosbag to .txt file for calibation")
    parser.add_argument("-b", "--bagfile", default= '/home/jake/rosbags/gp_spline/tag0.bag',type=str, help="ROS1 bagfile (*.bag) to read data from.")
    parser.add_argument("-o", "--outfile", type=str, default='/home/jake/ros_workspaces/gp_spline_ws/src/gp_spline/data/hw/in/hw.txt', help="Output path including a .txt file that will be used as a prefix (e.g. out.txt -> out_imu.txt etc.). Uses the original bagfile directory if none provided.")
    parser.add_argument("-s", "--start", default=0.0, type=float, help="Data acquistion start time (relative to when first apriltag is detected in rosbag)")
    parser.add_argument("-e", "--end", default=np.inf, type=float, help="Data acquistion end time (relative to when first apriltag is detected in rosbag)")
    parser.add_argument("-i", "--imu_ds_factor", default=1, type=int, help="Factor with which to downsample the imu messages")
    parser.add_argument("-a", "--apriltag_ds_factor", default=1, type=int, help="Factor with which to downsample the apriltag detection messages")
    args = vars(parser.parse_args())
    main(**args)