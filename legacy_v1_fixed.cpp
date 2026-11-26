// v1 (C++) 시뮬레이션 - 오류만 바로잡은 비교용 버전
// ------------------------------------------------------------
// 원래 코드에서 고친 점 (주석 [수정] 표시)
//  1) 삼중항 식에서 -kT*triplet 이 두 번 들어가 있었음 -> T->S 되돌아가는 항(kST)으로 바꿈
//  2) 단일항 식에서 T->S 되돌아오는 속도를 kT(반응 속도)로 썼음 -> 같은 전이 속도 kST 를 써야 대칭
//  3) 100단계 x dt 0.01 = 시간 1 만 계산 -> 라디칼 쌍이 거의 다 반응할 때까지(시간 10) 계산
//  4) '남아 있는 T 비율'이 아니라 '삼중항 경로로 만들어진 생성물의 비율'을 신호로 사용
//
// 그래도 남는 '모델' 한계 (코드 오류가 아니라 가정의 문제) -> v2 Python 에서 해결
//  * kST = 0.5 + 0.005*B : 자기장 '세기'가 클수록 전이가 빨라진다는 가정은 실제 라디칼 쌍과 맞지 않음
//  * 철새 나침반은 자기장의 '방향(기울기)'을 이용하는 것으로 알려져 있음 (Wiltschko 1972)
//  * 신호 확률 = 북쪽으로 이동할 확률 로 바로 연결 -> 감지/판단/비행 단계가 섞여 있음
//    (그래서 자기장이 0이어도 새가 북쪽으로 이동하는 이상한 결과가 나옴)
//
// 컴파일:  g++ -O2 -o legacy legacy_v1_fixed.cpp   실행: ./legacy
#include <iostream>
#include <cmath>
#include <random>

double calculateQuantumSignal(double magneticField) {
    double singlet = 1.0, triplet = 0.0;
    double prodS = 0.0, prodT = 0.0;       // [수정 4] 생성물 누적량
    const double dt = 0.01;
    const double kS = 1.0, kT = 1.0;
    const double kST = 0.5 + 0.005 * magneticField;   // 원래 가정 그대로 둠(한계)

    for (int i = 0; i < 1000; ++i) {       // [수정 3] 시간 10 까지
        double dSinglet = (-kS * singlet - kST * singlet + kST * triplet) * dt;   // [수정 2]
        double dTriplet = ( kST * singlet - kST * triplet - kT * triplet) * dt;   // [수정 1]
        prodS += kS * singlet * dt;
        prodT += kT * triplet * dt;
        singlet += dSinglet;
        triplet += dTriplet;
    }
    return prodT / (prodS + prodT);        // [수정 4]
}

struct AgentBird { double positionX, positionY; };

int main() {
    std::mt19937 randomEngine(5678);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (double B : {0.0, 50.0}) {
        AgentBird bird = {0.0, 0.0};
        double p = calculateQuantumSignal(B);
        for (int day = 1; day <= 10; ++day) {
            if (dist(randomEngine) < p) bird.positionY += 10.5;
            else bird.positionX += (dist(randomEngine) - 0.5) * 5.0;
        }
        std::cout << "B = " << B << " uT : 신호(삼중항 생성물 비율) = " << p
                  << ", 10일 뒤 위치 (" << bird.positionX << ", " << bird.positionY << ")\n";
    }
    std::cout << "-> 자기장 0 에서도 북쪽으로 이동: '신호 = 이동 확률' 연결이 생물학적으로 맞지 않음\n";
    return 0;
}
