#include <iostream>
#include <string>
#include <unistd.h>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/vui/vui_client.hpp>

static void usage() {
    std::cerr <<
        "Usage:\n"
        "  go2_grand_ICT_v2 move <vx> <vy> <vyaw>\n"
        "  go2_grand_ICT_v2 stop\n"
        "  go2_grand_ICT_v2 scrape\n"
        "  go2_grand_ICT_v2 sit\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return -1; }
    std::string cmd = argv[1];

    // DDS 초기화(eth0 고정)
    unitree::robot::ChannelFactory::Instance()->Init(0, "eth0");

    unitree::robot::go2::SportClient sport;
    sport.SetTimeout(0.2f);
    sport.Init();

    try {
        if (cmd == "move") {
            if (argc < 5) { usage(); return -2; }
            float vx = std::stof(argv[2]);
            float vy = std::stof(argv[3]);
            float vyaw = std::stof(argv[4]);
            return sport.Move(vx, vy, vyaw);
        } else if (cmd == "stop") {
            sport.Move(0.0f, 0.0f, 0.0f);
            return 0;
        } else if (cmd == "scrape") {
            sport.BalanceStand();    // 제스처 전 안정자세
            int rc = sport.Scrape(); // <- SDK에 존재하는 제스처 API
            sleep(3);
            return rc;
        } else if (cmd == "sit") {
            int rc = sport.Sit();    // <- SDK에 존재하는 앉기 API
            sleep(2);
            return rc;
        } else {
            usage();
            return -3;
        }
    } catch (const std::exception& e) {
        std::cerr << "[go2_grand_ICT_v2] exception: " << e.what() << std::endl;
        return -10;
    }
}
