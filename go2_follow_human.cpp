#include <iostream>
#include <cstdlib>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/vui/vui_client.hpp>

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: go2_follow_human vx vy vyaw" << std::endl;
        return -1;
    }

    float vx = std::stof(argv[1]);
    float vy = std::stof(argv[2]);
    float vyaw = std::stof(argv[3]);

    unitree::robot::ChannelFactory::Instance()->Init(0, "eth0");

    unitree::robot::go2::SportClient sport;
    sport.SetTimeout(0.1f);
    sport.Init();

    // optional: light off
    unitree::robot::go2::VuiClient vui;
    vui.SetBrightness(0);

    // ✅ 3개 인자 호출
    sport.Move(vx, vy, vyaw);

    return 0;
}
