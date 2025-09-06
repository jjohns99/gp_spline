import sqlite3

from yaml import parse
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation as R
import argparse
import os
import numpy as np
import navpy
import csv


class BagFileParser():
    """
    A simple parser for ROS2 bags using sqlite. This parser will convert the rosbag into a .txt file format that can 
    then be used with the estimators in this package offline. Created for use with ROS Foxy which lacks rosbag2_py.

    Bag should contain the following sensor measurements:
        RTK GPS relative measurements (ublox_read_2/msg/RelPos)
        GPS velocity measurements for the rover (ublox_read_2/msg/PosVelEcef)
        GPS velocity measurements for the base (ublox_read_2/msg/PosVelEcef)
        Camera to apriltag relative pose measurements (apriltag_msgs/msg/AprilTagDetectionArray)
        Imu measurements (sensor_msgs/msg/Imu)
        Imu base measurements (sensor_msgs/msg/Imu)
        Ground truth for rover to base measurements (nav_msgs/msg/Odometry) - Motion capture only.
    
    WARNING: Ensure your workspace has sourced ONLY your ROS2 disto. Otherwise this parser may confuse message types 
    with the ROS1 equivalents
    """
    def __init__(self, bagfile):
        self.conn = sqlite3.connect(bagfile)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of:type_of for id_of,name_of,type_of in topics_data}
        self.topic_id = {name_of:id_of for id_of,name_of,type_of in topics_data}
        self.topic_msg_message = {name_of:get_message(type_of) for id_of,name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    def get_messages(self, topic_name):

        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        # Deserialise all and timestamp them
        return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]

def get_px4_ros_clock_offset(timesync_msgs):
    avg_offset = 0.0
    count = 0

    for msg in timesync_msgs:
        avg_offset += msg.observed_offset * 1E-6

def main(bagfile, mocap, start, end, outfile=None):
    parser = BagFileParser(bagfile)

    apriltag = parser.get_messages('/apriltag_pose')
    imu = parser.get_messages('/fmu/out/sensor_combined')
    rover_truth = parser.get_messages('/rhodey/pose_ned')
    april0 = parser.get_messages('/april0/pose_ned')
    april1 = parser.get_messages('/april1/pose_ned')
    april2 = parser.get_messages('/april2/pose_ned')
    april3 = parser.get_messages('/april3/pose_ned')

    msgs = [april0, april1, april2, april3, apriltag, imu]
    t0 = []

    

    # Find t0
    for msg in msgs:
        try:
            t0.append(msg[0][1].header.stamp.sec + msg[0][1].header.stamp.nanosec * 1E-9)
        except:
            print(msg[0])
            if len(msg):
                t0.append(msg[0][1].timestamp*1E-6)

    # if abs(t0[-1] - t0[-2]) > 100:
    #     offset = get_offset(timesync)

    first_time = min(t0)

    if not outfile:
        outpath = os.path.join("/", *bagfile.split('/')[:-1], "out")
    else:
        assert outfile.split('.')[-1] == 'txt', f"Provided path must end with a .txt file, got: {outfile}"
        outpath = os.path.join(*outfile.split('.')[:-1])

    comb = open(outpath + "_combined.txt", "w")
    with open(outpath + "_apriltag.txt", 'w') as f:

        # Apriltag (t(s), translation(m): x, y, z, rotation(quat): w, x, y, z)
        for msg in apriltag:
            if len(msg[1].detections):
                stamp = msg[1].header.stamp
                time = stamp.sec + stamp.nanosec * 1E-9 - first_time
                if time < start or time > end:
                    continue
                for d in msg[1].detections:
                    position = d.pose.pose.pose.position
                    orientation = d.pose.pose.pose.orientation

                    f.write("{} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
                        d.id, time, position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z))
                    f.write("\n")
                    
                    comb.write("CAM {} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
                        d.id, time, position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z))
                    comb.write("\n")

    
    with open(outpath + "_rover_truth.txt", 'w') as f:

        # Motion capture truth (t(s), translation(m): x, y, z, rotation(quat): w, x, y, z)
        for msg in rover_truth:
            stamp = msg[1].header.stamp
            time = stamp.sec + stamp.nanosec * 1E-9 - first_time
            if time >= 0 and (time >= start and time < end):
                position = msg[1].pose.position
                orientation = msg[1].pose.orientation

                f.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
                    time, position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z))
                f.write("\n")

    with open(outpath + "_april0.txt", 'w') as f:

        # Motion capture truth (t(s), translation(m): x, y, z, rotation(quat): w, x, y, z)
        for msg in april0:
            stamp = msg[1].header.stamp
            time = stamp.sec + stamp.nanosec * 1E-9 - first_time
            if time >= 0 and (time >= start and time < end):
                position = msg[1].pose.position
                orientation = msg[1].pose.orientation

                f.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
                    time, position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z))
                f.write("\n")

    with open(outpath + "_april1.txt", 'w') as f:

        # Motion capture truth (t(s), translation(m): x, y, z, rotation(quat): w, x, y, z)
        for msg in april1:
            stamp = msg[1].header.stamp
            time = stamp.sec + stamp.nanosec * 1E-9 - first_time
            if time >= 0 and (time >= start and time < end):
                position = msg[1].pose.position
                orientation = msg[1].pose.orientation

                f.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
                    time, position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z))
                f.write("\n")

    with open(outpath + "_april2.txt", 'w') as f:

        # Motion capture truth (t(s), translation(m): x, y, z, rotation(quat): w, x, y, z)
        for msg in april2:
            stamp = msg[1].header.stamp
            time = stamp.sec + stamp.nanosec * 1E-9 - first_time
            if time >= 0 and (time >= start and time < end):
                position = msg[1].pose.position
                orientation = msg[1].pose.orientation

                f.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
                    time, position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z))
                f.write("\n")

    with open(outpath + "_april3.txt", 'w') as f:

        # Motion capture truth (t(s), translation(m): x, y, z, rotation(quat): w, x, y, z)
        for msg in april3:
            stamp = msg[1].header.stamp
            time = stamp.sec + stamp.nanosec * 1E-9 - first_time
            if time >= 0 and (time >= start and time < end):
                position = msg[1].pose.position
                orientation = msg[1].pose.orientation

                f.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
                    time, position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z))
                f.write("\n")




    with open(outpath + "_imu.txt", 'w') as f:

        # inertial measurement unit (t(s), linear accel(m/s^2): x, y, z, angular vel(rad/s): x, y, z)
        for msg in imu:
            time = msg[1].timestamp*1E-6 - first_time
            if time >= 0 and (time >= start and time < end):
                accel = msg[1].accelerometer_m_s2
                omega = msg[1].gyro_rad

                f.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(time, accel[0], accel[1], accel[2], omega[0], omega[1], omega[2]))
                f.write("\n")
                
                comb.write("IMU {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(time, accel[0], accel[1], accel[2], omega[0], omega[1], omega[2]))
                comb.write("\n")


    print(f"Data written to {outpath}")
    comb.close()

    # Sort the combined data
    comb = open(outpath + "_combined.txt", "r")
    csv1 = csv.reader(comb, delimiter=" ")
    sort = sorted(csv1, key=lambda x : float(x[1]))

    comb_sort = open(outpath+"_sort.txt", "w")
    for line in sort:
        str1 = " "
        comb_sort.write(str1.join(line))
        comb_sort.write("\n")

    comb_sort.close()
    comb.close()





if __name__ == "__main__":
    # Parse through arguments
    parser = argparse.ArgumentParser(description="Convert a ROS2 rosbag to .txt file for calibation")
    parser.add_argument("-b", "--bagfile", default= '/home/jake/rosbags/gp_spline/manual/jake_05-09-2023_15:18:26_0.db3',type=str, help="ROS2 bagfile (*.db3) to read data from.")
    parser.add_argument("-o", "--outfile", type=str, default='/home/jake/ros_workspaces/gp_spline_ws/src/gp_spline/data/hw/in/hw.txt', help="Output path including a .txt file that will be used as a prefix (e.g. out.txt -> out_imu.txt etc.). Uses the original bagfile directory if none provided.")
    parser.add_argument("-m", "--mocap", default=True, type=bool, help="Flag for calibration in mocap. Determines whether ground truth is recorded")
    parser.add_argument("-s", "--start", default=0.0, type=float, help="Data acquistion start time (relative to when first apriltag is detected in rosbag)")
    parser.add_argument("-e", "--end", default=np.inf, type=float, help="Data acquistion end time (relative to when first apriltag is detected in rosbag)")
    args = vars(parser.parse_args())
    main(**args)
