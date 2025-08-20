#include <iostream>
#include <csignal>
#include <unistd.h>

#include <unitree/robot/go2/sport/sport_client.hpp>

bool stopped = false;

void sigint_handler(int sig)
{
    if (sig == SIGINT) {
        stopped = true;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        return -1;
    }

    // DDS 초기화 (eth0 같은 네트워크 인터페이스 전달 필요)
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

    // SportClient 객체 생성
    unitree::robot::go2::SportClient sport_client;
    sport_client.SetTimeout(10.0f); 
    sport_client.Init();

    // Ctrl+C 시그널 처리
    signal(SIGINT, sigint_handler);

    std::cout << "👉 Step 1: Hello 실행" << std::endl;
    sport_client.Hello();
    sleep(5);  // 동작 관찰용 대기

    if (stopped) return 0;

    std::cout << "👉 Step 2: BalanceStand 실행" << std::endl;
    sport_client.BalanceStand();
    sleep(5);

    if (stopped) return 0;

    std::cout << "👉 Step 3: StandDown 실행" << std::endl;
    sport_client.StandDown();
    sleep(5);

    if (stopped) return 0;

    std::cout << "👉 Step 4: RecoveryStand 실행" << std::endl;
    sport_client.RecoveryStand();
    sleep(5);

    if (stopped) return 0;

    std::cout << "👉 Step 5: Content 실행" << std::endl;
    sport_client.Content();
    sleep(5);

    if (stopped) return 0;

    std::cout << "👉 Step 6: Heart 실행" << std::endl;
    sport_client.Heart();
    sleep(5);

    if (stopped) return 0;

    std::cout << "👉 Step 7: Pose 실행" << std::endl;
    sport_client.Pose(true);
    sleep(5);
    sport_client.Pose(false);

    if (stopped) return 0;

    std::cout << "👉 Step 8: Scrape 실행" << std::endl;
    sport_client.Scrape();
    sleep(5);

    if (stopped) return 0;

    std::cout << "👉 Step 9: Sit 실행" << std::endl;
    sport_client.Sit();
    sleep(5);

    if (stopped) return 0;

    std::cout << "👉 Step 10: RiseSit 실행" << std::endl;
    sport_client.RiseSit();
    sleep(5);


    std::cout << "✅ 모든 동작이 완료되었습니다." << std::endl;
    return 0;
}
