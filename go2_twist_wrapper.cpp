#include <unitree/robot/go2/sport/sport_client.hpp>
#include <iostream>
#include <cstdlib>

int main(int argc, char **argv)
{
  if (argc < 4)
  {
    std::cout << "Usage: go2_twist_wrapper vx vy vyaw" << std::endl;
    return -1;
  }

  float vx = std::stof(argv[1]);
  float vy = std::stof(argv[2]);
  float vyaw = std::stof(argv[3]);

  // Network interface는 고정 (eth0 또는 enp3s0)
  unitree::robot::ChannelFactory::Instance()->Init(0, "eth0");

  unitree::robot::go2::SportClient sport;
  sport.Init();
  sport.Move(vx, vy, vyaw);

  return 0;
}
