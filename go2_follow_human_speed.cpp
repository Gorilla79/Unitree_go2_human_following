#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <iostream>

int main(int argc, char** argv) {
  const char* nic = (argc >= 2 ? argv[1] : "eth0");  // allow override
  unitree::robot::ChannelFactory::Instance()->Init(0, nic);

  unitree::robot::go2::SportClient sport;
  sport.SetTimeout(1.0f);
  sport.Init();
  // sport.WaitLeaseApplied(); // uncomment if your setup enables leases

  // Make sure we can move
  sport.BalanceStand();

  std::cerr << "Go2 control server ready on " << nic << ".\n"
            << "Send lines: vx vy vyaw (m/s, m/s, rad/s). Ctrl+C to quit.\n";

  while (std::cin) {
    float vx, vy, vyaw;
    if (!(std::cin >> vx >> vy >> vyaw)) break;
    sport.Move(vx, vy, vyaw);
  }
  sport.StopMove();
  return 0;
}
