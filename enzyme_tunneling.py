"""
효소 속 수소 터널링 시뮬레이션  (열역학 + 양자역학)
====================================================

[생명 현상]
  콩(대두)의 효소 '리폭시게네이스-1(SLO-1)'은 지방산에서 수소를 떼어 내는 반응을 촉매한다.
  수소(H)를 무거운 동위원소 중수소(D)로 바꾸면 반응이 약 80배 느려진다(실험값 kH/kD ≈ 80, 25~30 °C).
  고전적인 화학(활성화 에너지 + 영점 에너지 차이)으로 설명 가능한 최대치는 약 7 정도라서,
  '수소가 에너지 언덕을 넘지 않고 뚫고 지나간다(양자 터널링)'는 설명이 받아들여지고 있다.
  (Klinman 연구팀, 예: Meyer & Klinman 2008 PNAS; Klinman & Kohen 2013 Annu. Rev. Biochem.)

[고등학교 화학과 연결]
  * 활성화 에너지 Ea : 반응이 일어나려면 넘어야 하는 에너지 언덕
  * 온도가 높을수록 언덕을 넘을 만큼 빠른 분자의 비율이 커진다(볼츠만 분포) -> 반응 속도 증가
  * 아레니우스 식 k = A·exp(-Ea / RT)  ->  ln k 와 1/T 는 직선 관계
  * 양자역학에서는 입자가 '파동'이므로 언덕보다 에너지가 낮아도 일정 확률로 통과(터널링)할 수 있다
    -> 가벼운 입자(H)일수록, 언덕이 얇을수록 잘 통과

[계산 방법]
  * 에너지 언덕 모양: V(x) = V0 / cosh^2(x/a)  (에카르트형 언덕, 투과 확률 공식이 정확히 알려져 있음)
  * 투과 확률 P(E) (Landau-Lifshitz 양자역학 교과서 공식)
        P = sinh^2(π k a) / [ sinh^2(π k a) + cosh^2( (π/2)·sqrt(8 m V0 a^2/ħ^2 - 1) ) ],  k = sqrt(2mE)/ħ
  * 반응 속도 ∝ ∫ P(E) · exp(-E/kT) dE   (모든 에너지의 분자에 대해 '통과 확률 x 그 에너지를 가질 확률'을 더함)
  * 고전 계산: P(E) = 0 (E < V0), 1 (E >= V0)  -> 아레니우스 식과 같아짐
  * 영점 에너지(ZPE): C-H 결합 진동의 바닥 에너지. 반응 출발점이 ZPE만큼 높아서 실제 언덕은 V0 - ZPE.
        C-H 진동수 약 2900 cm^-1 -> ZPE ≈ 0.180 eV,  C-D 는 질량 때문에 약 1/sqrt(2) -> ≈ 0.127 eV

[정직하게 말해야 할 점]
  * 언덕 높이·폭은 실제 SLO-1 값을 '맞춘' 것이 아니라 같은 자릿수의 시연용 값이다.
  * 실제 효소에서는 단백질의 진동이 수소 주개와 받개 사이 거리를 순간적으로 좁혀 주는 효과가 중요하다는
    모델(전자-양성자 결합, '거리 표집')이 쓰인다. 여기서는 그 효과를 '언덕 폭 a'로만 단순화했다.

실행:  python enzyme_tunneling.py          (그래프 창)
       python enzyme_tunneling.py --save   (figures/ 폴더에 PNG 저장)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

HBAR = 1.054571817e-34      # J·s
KB = 1.380649e-23           # J/K
H_PLANCK = 6.62607015e-34   # J·s
EV = 1.602176634e-19        # J
AMU = 1.66053907e-27        # kg

V0_EV = 0.60                # 영점 에너지를 빼기 전 언덕 높이 (eV) - 시연용
WIDTH_A = 0.20e-10          # 언덕 폭 매개변수 a (m) = 0.20 Å - 시연용
ZPE = {"H": 0.180, "D": 0.127}          # eV
MASS = {"H": 1.008 * AMU, "D": 2.014 * AMU}


def transmission(E_J, V_J, m, a=WIDTH_A):
    """에카르트형 언덕 V/cosh^2(x/a) 의 정확한 양자 투과 확률."""
    k = np.sqrt(2 * m * np.maximum(E_J, 1e-30)) / HBAR
    arg = 8 * m * V_J * a**2 / HBAR**2 - 1
    c = np.cosh(np.pi / 2 * np.sqrt(arg)) if arg > 0 else np.cos(np.pi / 2 * np.sqrt(-arg))
    s = np.sinh(np.pi * k * a)
    # 큰 수 오버플로 방지: 비율 형태로 계산
    with np.errstate(over="ignore", invalid="ignore"):
        P = 1.0 / (1.0 + (c / s) ** 2)
    return np.nan_to_num(P, nan=0.0)


def rate(T, iso, quantum=True, a=WIDTH_A):
    """반응 속도 상수 (1/s). 고전: 아이링 식 (kT/h)·exp(-V/kT) 와 같다."""
    V = (V0_EV - ZPE[iso]) * EV
    kT = KB * T
    pref = kT / H_PLANCK
    if not quantum:
        return pref * np.exp(-V / kT)
    E = np.linspace(1e-4, 3.0, 30000) * V          # 0 ~ 3V 범위 적분
    P = transmission(E, V, MASS[iso], a)
    integrand = P * np.exp(-E / kT)
    trap = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.0 미만 호환
    return pref * trap(integrand, E) / kT


def arrhenius_Ea(Ts, ks):
    """ln k 대 1/T 기울기로 겉보기 활성화 에너지 (kcal/mol)."""
    slope = np.polyfit(1 / np.asarray(Ts), np.log(ks), 1)[0]
    return -slope * KB / 4184 * 6.02214076e23


def setup_font():
    from matplotlib import font_manager
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        try:
            font_manager.findfont(f, fallback_to_default=False); plt.rcParams["font.family"] = f; break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False


def fig_barrier():
    """언덕 그림 + 에너지별 통과 확률."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.linspace(-0.6, 0.6, 400)
    V = (V0_EV - ZPE["H"]) / np.cosh(x * 1e-10 / WIDTH_A) ** 2
    a1.fill_between(x, V, color="#f5cba7"); a1.plot(x, V, color="#a04000")
    a1.axhline(0.15, color="#2471a3", ls="--"); a1.text(-0.58, 0.165, "수소의 에너지 (언덕보다 낮음)", color="#2471a3")
    a1.annotate("", xy=(0.45, 0.15), xytext=(-0.45, 0.15), arrowprops=dict(arrowstyle="->", color="#2471a3", lw=2))
    a1.text(0.20, 0.33, "고전: 못 넘음\n양자: 일부가 '뚫고' 지나감", ha="left", fontsize=10)
    a1.set_xlabel("반응 좌표 (Å)"); a1.set_ylabel("에너지 (eV)"); a1.set_title("에너지 언덕(활성화 에너지)")
    E = np.linspace(0.01, 0.8, 400)
    for iso, col in [("H", "#c0392b"), ("D", "#2471a3")]:
        Vb = (V0_EV - ZPE[iso]) * EV
        a2.semilogy(E, transmission(E * EV, Vb, MASS[iso]), color=col, label=f"{iso} ({'수소' if iso=='H' else '중수소'})")
        a2.axvline(V0_EV - ZPE[iso], color=col, ls=":", lw=1)
    a2.set_ylim(1e-12, 2); a2.set_xlabel("입자의 에너지 (eV)"); a2.set_ylabel("통과 확률 (로그)")
    a2.set_title("가벼운 H가 훨씬 잘 뚫고 지나간다\n(점선: 각 언덕 높이 = 고전적으로 넘는 기준)")
    a2.legend(); a2.grid(alpha=.25, which="both"); fig.tight_layout()
    return fig


def fig_arrhenius():
    Ts = np.linspace(278, 323, 10)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for iso, col in [("H", "#c0392b"), ("D", "#2471a3")]:
        kq = [rate(T, iso, True) for T in Ts]; kc = [rate(T, iso, False) for T in Ts]
        a1.plot(1000 / Ts, np.log(kq), "-o", color=col, label=f"{iso} 양자(터널링 포함)")
        a1.plot(1000 / Ts, np.log(kc), "--", color=col, alpha=.6, label=f"{iso} 고전")
    a1.set_xlabel("1000 / T (1/K)"); a1.set_ylabel("ln k"); a1.set_title("아레니우스 그래프")
    a1.legend(fontsize=8); a1.grid(alpha=.25)
    Tc = Ts - 273.15
    kie_q = [rate(T, "H") / rate(T, "D") for T in Ts]
    kie_c = [rate(T, "H", False) / rate(T, "D", False) for T in Ts]
    a2.plot(Tc, kie_q, "-o", color="#7d3c98", label="양자 모델")
    a2.plot(Tc, kie_c, "--", color="gray", label="고전 모델(영점 에너지만)")
    a2.axhspan(76, 86, color="#f9e79f", alpha=.6, label="SLO-1 실험값 ≈ 81±5")
    a2.set_yscale("log"); a2.set_xlabel("온도 (°C)"); a2.set_ylabel("kH / kD (동위원소 효과)")
    a2.set_title("H를 D로 바꾸면 얼마나 느려지나?"); a2.legend(fontsize=8); a2.grid(alpha=.25, which="both")
    fig.tight_layout()
    return fig


def main():
    setup_font()
    Ts = np.linspace(278, 323, 10)
    print("== 25 °C (298 K) 결과 ==")
    for iso in ("H", "D"):
        kq, kc = rate(298, iso), rate(298, iso, False)
        print(f"  {iso}: 고전 k = {kc:.3e} /s, 양자 k = {kq:.3e} /s, 터널링 덕분에 {kq/kc:.1f}배")
    print(f"  KIE 고전 = {rate(298,'H',False)/rate(298,'D',False):.1f},  양자 = {rate(298,'H')/rate(298,'D'):.1f}")
    for iso in ("H", "D"):
        print(f"  겉보기 활성화 에너지 {iso}: 고전 {arrhenius_Ea(Ts,[rate(T,iso,False) for T in Ts]):.1f} kcal/mol, "
              f"양자 {arrhenius_Ea(Ts,[rate(T,iso) for T in Ts]):.1f} kcal/mol")
    figs = {"tun_barrier": fig_barrier(), "tun_arrhenius": fig_arrhenius()}
    if "--save" in sys.argv:
        import os; os.makedirs("figures", exist_ok=True)
        for k, f in figs.items():
            f.savefig(f"figures/{k}.png", dpi=200)
        print("figures/ 폴더에 저장 완료")
    else:
        plt.show()


if __name__ == "__main__":
    main()
