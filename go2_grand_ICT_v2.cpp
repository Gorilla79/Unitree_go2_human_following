// example/go2/go2_grand_ICT_v2.cpp
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/vui/vui_client.hpp>

static void usage() {
    std::cerr
        << "Usage:\n"
        << "  go2_grand_ICT_v2 move <vx> <vy> <vyaw>\n"
        << "  go2_grand_ICT_v2 stop\n"
        << "  go2_grand_ICT_v2 hello\n"
        << "  go2_grand_ICT_v2 damp\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return -1; }
    std::string cmd = argv[1];

    // 네트워크 초기화 (eth0 고정)
    unitree::robot::ChannelFactory::Instance()->Init(0, "eth0");

    unitree::robot::go2::SportClient sport;
    sport.SetTimeout(0.1f);
    sport.Init();

    unitree::robot::go2::VuiClient vui; // 옵션: LED, 음성 등
    //vui.SetBrightness(0); // 원하면 사용

    try {
        if (cmd == "move") {
            if (argc < 5) { usage(); return -2; }
            float vx   = std::stof(argv[2]);
            float vy   = std::stof(argv[3]);
            float vyaw = std::stof(argv[4]);
            // 이동
            sport.Move(vx, vy, vyaw);
            return 0;
        } else if (cmd == "stop") {
            // 간단 정지: 속도 0
            sport.Move(0.0f, 0.0f, 0.0f);
            return 0;
        } else if (cmd == "hello") {
            // SDK 문서 기준 Hello() 지원
            int ret = sport.Hello();
            return ret == 0 ? 0 : ret;
        } else if (cmd == "damp") {
            int ret = sport.Damp();
            return ret == 0 ? 0 : ret;
        } else {
            usage();
            return -3;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: exception: " << e.what() << std::endl;
        return -4;
    }
}
