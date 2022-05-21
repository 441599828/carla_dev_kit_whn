# Carla data collection  in intersection
terminal 1 :cd opt/carla-sim/
            bash CarlaUE4.sh

terminal 2 :python generate_traffic.py -n 50 -w 50 -b 0 --safe

terminal 3 :python set_weather.py

# for data collection

terminal 4 :python collect_data.py

change the depth image into kitti depth image format: python dep2kitti.py

calculate datasets' means & std and maxmindepth: python calculate_data.py

generate train val test data into .txt file: python generate_txtfile.py

# for map generate

generate static_map from cameras: python generate_pngfile.py

generate static_map from lidars: python generate_map.py

copy static_map for every train frame: python copymap2frame.py

other scripts: 

vis lidar in open3d visualizer: python vis_lidar.py

generate gt camera dep seg rgb: python generate_pngfile.py

convertor: pcd2bin.py pcd2depth.py etc.

# Other scripts:

test_camera_effect.py: test f-value and focal distance in the camera
