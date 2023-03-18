sensor_msgs/Image tracer_image
sensor_msgs/Image traced_image
sensor_msgs/Image tracer_depth_image
sensor_msgs/Image traced_depth_image
string tracer_robot # e.g. "/tb3_0"
string traced_robot
float64 src_map_origin_x
float64 src_map_origin_y
float64 dst_map_origin_x
float64 dst_map_origin_y
geometry_msgs/Pose arrived_pose
bool aborted
int64 stamp