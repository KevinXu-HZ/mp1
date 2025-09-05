#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from models.simple_enet import SimpleENet
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError

from utils.Line import Line
from utils.util import perspective_transform, quaternion_to_euler
from utils.line_fit import line_fit, tune_fit, bird_fit, final_viz
from utils.ground_truth_generator import GroundTruthGenerator


class LaneVisualizer(Node):
    def __init__(self, ckpt_fn: str):
        super().__init__('lane_visualizer')

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        
        try:
            checkpoint = torch.load(ckpt_fn, map_location='cpu')
            self.model = SimpleENet(num_classes=2).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.get_logger().info(f"Model loaded successfully from {ckpt_fn}")
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")
            sys.exit()
            
        # Load BEV configuration
        config_path = os.path.join(os.path.dirname(__file__), '../data/bev_config.json')
        if not os.path.exists(config_path):
            config_path = 'bev_config.json'
        try:
            with open(config_path, 'r') as f:
                self.bev_config = json.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"BEV config file not found at {config_path}")
            sys.exit()
            
        self.raw_img = None
        self.bridge = CvBridge()
        
        # QoS profile for image transport
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscriber
        self.camera_sub = self.create_subscription(
            Image,
            '/front_camera/image_raw',
            self.camera_callback,
            qos_profile
        )
        
        # Publishers
        self.final_pub = self.create_publisher(Image, 'visualize_lanes/final_lane', 1)
        self.poly_pub = self.create_publisher(Float32MultiArray, 'lane_fit/poly', 1)

        self.lane_rgba = (0, 255, 0), 0.5
        
        # Config for polyfit
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.hist = False
        self.detected = False

        # Ground truth mask publisher
        world = os.path.join(os.path.dirname(__file__), '..', '..', 'gem_simulator/gem_gazebo/worlds/smaller_track.world')
        self.ground_truth_gen = GroundTruthGenerator(world, 30)
        self.model_state_sub = self.create_subscription(
            ModelStates,
            '/model_states',
            self.listener_callback,
            10
        )
        
        # Timer for processing loop
        self.timer = self.create_timer(0.02, self.process_callback)  # 50 Hz

        self.get_logger().info('Lane visualizer node initialized')

    def camera_callback(self, msg):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.raw_img = cv_image.copy()
            print(self.raw_img.shape)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    def listener_callback(self, msg):
        try:
            index = msg.name.index('gem')
            pose = msg.pose[index]
            x = pose.position.x
            y = pose.position.y
            q = pose.orientation
            _, _, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
            self.cur_gem_pose = [x, y], yaw
        except ValueError:
            self.get_logger().info('Model not found.')

    def process_callback(self):
        if self.raw_img is None:
            self.get_logger().info('Raw image is not found.')
            return
        try:
            mask = self.get_segmentation_mask(self.raw_img)
            combine_fit_img, bird_fit_img, ret = self.fit_poly_lanes(mask)
            if self.detected and ret is not None:
                self.final_pub.publish(self.bridge.cv2_to_imgmsg(combine_fit_img, '8UC3'))
                center_poly = (np.add(ret["left_fit"], ret["right_fit"]) / 2).tolist()
                poly_msg = Float32MultiArray()
                poly_msg.data = center_poly
                self.poly_pub.publish(poly_msg)
        except Exception as e:
            self.get_logger().error(f"Error in processing callback: {e}")

    def get_segmentation_mask(self, img):
        """
        Process an input image to generate a binary lane segmentation mask.

        Args:
            img (numpy.ndarray): Input image in BGR format.

        Returns:
            binary_output (numpy.ndarray): Binary image (uint8) of the same size as input, where lane pixels are 255 and others are 0.

        Steps:
            1. Resize the image to the model input size (640x384).
            2. Convert to grayscale and normalize pixel values to [0, 1] as type float32.
            3. Convert the normalized image to a PyTorch tensor and add channel and batch dimensions.
            4. Run the model to generate class prediction.
            5. Convert the predicted mask to a binary format and resize it to the original image size.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot proceed with detection.")

        img_resized = None
        img_grayscale = None
        img_normalized = None

        ##### Your code starts here #####
        
        ##### Your code ends here #####

        # Step 3: Convert to tensor and add channel/batch dimension
        input_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0)  # Add channel dimension
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        # Step 4: Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            pred_mask = torch.argmax(output, dim=1)
            # probabilities = torch.softmax(output, dim=1)
        binary_output = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        ##### Your code starts here #####

        ##### Your code ends here #####

        return binary_output
    
    def fit_poly_lanes(self, binary_img):
        binary_warped, M, Minv = perspective_transform(binary_img, np.float32(self.bev_config["src"]))
        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(binary_warped)
            if ret is None:
                self.get_logger().debug("ret is None; returning None for both.")
                self.detected = False
                return None, None, None
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            self.detected = True
        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(binary_warped)
                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)
                    self.detected = True
            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(binary_warped, left_fit, right_fit)
                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)
                    self.detected = True
                else:
                    self.detected = False
        bird_fit_img = None
        combine_fit_img = None
        if ret is not None:
            self.get_logger().debug("Model detected lanes")
            bird_fit_img = bird_fit(binary_warped, ret, Minv)
            combine_fit_img = final_viz(self.raw_img, left_fit, right_fit, Minv)
            self.detected = True
        else:
            self.get_logger().debug("Model unable to detect lanes")
        # return polynomial and save the images as a class member variables?
        return combine_fit_img, bird_fit_img, ret


def main(args=None):
    rclpy.init(args=args)
    
    # Default checkpoint path - should be updated based on actual path
    default_ckpt = os.path.join(
        os.path.dirname(__file__), 
        '..', 'checkpoints', 'simple_enet_checkpoint_epoch_40.pth'
    )
    
    try:
        visualizer = LaneVisualizer(ckpt_fn=default_ckpt, debug=False)
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'visualizer' in locals():
            visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
