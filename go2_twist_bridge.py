
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import subprocess

class Go2TwistBridge(Node):
    def __init__(self):
        super().__init__('go2_twist_bridge')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

    def cmd_vel_callback(self, msg: Twist):
        vx = msg.linear.x
        vy = msg.linear.y
        vyaw = msg.angular.z

        # float → string으로 변환
        args = [f"{vx:.3f}", f"{vy:.3f}", f"{vyaw:.3f}"]

        self.get_logger().info(f'[cmd_vel] vx: {vx:.2f}, vy: {vy:.2f}, vyaw: {vyaw:.2f}')
        subprocess.run(["sudo", "/home/unitree/unitree_sdk2-main/build/bin/go2_twist_wrapper"] + args)


def main(args=None):
    rclpy.init(args=args)
    node = Go2TwistBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
