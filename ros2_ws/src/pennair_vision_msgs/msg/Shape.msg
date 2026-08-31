# One detected shape in a single frame. The frame's stamp lives in the
# enclosing ShapeArray header.

# Centroid in pixels from the frame center, x right, y up (matches the
# camera intrinsics, whose principal point is (0, 0)).
int32 x
int32 y

# Depth of the flat surface along the optical axis, inches.
# -1 until the reference circle has been seen once.
float64 depth

# Centroid in the camera frame, inches: Z * K^-1 * [x, y, 1].
geometry_msgs/Point position

# Whether this shape is the reference circle used for depth.
bool is_circle

# Outline vertices in image pixels (top-left origin), z unused.
geometry_msgs/Polygon outline
