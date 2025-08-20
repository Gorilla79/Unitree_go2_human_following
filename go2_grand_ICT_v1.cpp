#include <iostream>
#include <string>
#include <cmath>
#include <unistd.h>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

static float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage:\n"
                  << "  " << argv[0] << " <netIf> <action> [args]\n"
                  << "  actions:\n"
                  << "    move <vx> <vy> <vyaw>   # m/s, m/s, rad/s\n"
                  << "    stop                    # StopMove()\n"
                  << "    hello                   # Hello()\n"
                  << "    balance                 # BalanceStand()\n"
                  << "    stand                   # StandUp()\n"
                  << "    speed <level>           # -1 slow, 0 normal, 1 fast\n";
        return -1;
    }

    std::string net_if = argv[1];
    std::string action = argv[2];

    // DDS 초기화
    unitree::robot::ChannelFactory::Instance()->Init(0, net_if.c_str());

    unitree::robot::go2::SportClient sport;
    sport.SetTimeout(0.1f);  // 실시간성 높게 (환경에 따라 0.3~1.0f로 조정)
    sport.Init();            // <-- 반환형 void 이므로 대입 금지!

    auto safe_stop = [&]() {
        sport.StopMove();
        usleep(200 * 1000);
    };

    if (action == "move") {
        if (argc < 6) {
            std::cerr << "move requires vx vy vyaw\n";
            return -3;
        }
        float vx   = std::stof(argv[3]);
        float vy   = std::stof(argv[4]);
        float vyaw = std::stof(argv[5]);

        // 안전 클램프(문서 권장 범위 가정)
        vx   = clampf(vx,   -2.5f, 3.8f);
        vy   = clampf(vy,   -1.0f, 1.0f);
        vyaw = clampf(vyaw, -4.0f, 4.0f);

        int ret = sport.Move(vx, vy, vyaw);
        if (ret != 0) {
            std::cerr << "Move ret=" << ret << std::endl;
            return ret;
        }
        // Move는 최신 명령이 일정 시간 유지됨(외부 루프에서 주기 호출 권장)

    } else if (action == "stop") {
        sport.StopMove();

    } else if (action == "hello") {
        int ret = sport.Hello();
        if (ret != 0) {
            std::cerr << "Hello ret=" << ret << std::endl;
            return ret;
        }

    } else if (action == "balance") {
        sport.BalanceStand();

    } else if (action == "stand") {
        sport.StandUp();

    } else if (action == "speed") {
        if (argc < 4) {
            std::cerr << "speed requires level (-1/0/1)\n";
            return -4;
        }
        int level = std::stoi(argv[3]);
        if (level < -1) level = -1;
        if (level >  1) level =  1;
        int ret = sport.SpeedLevel(level);
        if (ret != 0) {
            std::cerr << "SpeedLevel ret=" << ret << std::endl;
            return ret;
        }

    } else {
        std::cerr << "Unknown action: " << action << std::endl;
        return -5;
    }

    return 0;
}
