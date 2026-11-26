"""
라디칼 쌍 나침반 시뮬레이션 v2  (양자 스핀 계산 + 철새 비행)
==============================================================

[이전 버전과의 차이]
  v1 (C++)    : "자기장이 세지면 전이율이 커진다"고 직접 정한 식 + 식에 오타(삼중항 항 중복)
  v1.5 (Python): 속도식(미분방정식)은 바로잡았지만 여전히 '세기'만 다룸
  v2 (이 파일) : 전자 2개 + 원자핵 1개의 스핀을 양자역학으로 직접 계산
                 -> 자기장의 '세기'가 아니라 '방향(각도)'에 따라 결과가 달라짐을 보여줌
                 -> 그 신호로 철새가 방향을 고르는 과정을 단계별로 분리

[무엇을 계산하나]
  * 라디칼 쌍: 전자 A(원자핵 1개와 상호작용), 전자 B(원자핵 없음) -> 상태 수 2x2x2 = 8
  * 해밀토니안 H = 제이만 상호작용(전자-외부 자기장) + 초미세 상호작용(전자-원자핵)
  * 초미세 상호작용은 '축 방향' 이방성(Azz만 있음) -> 분자에 대한 자기장 방향에 따라 결과가 달라짐
  * 단일항 수득률 Φ_S = (1/M) Σ |<m|P_S|n>|^2 · k^2 / (k^2 + ω_mn^2)
      (Timmel 등 1998, Ritz 등 2000에서 쓰는 교과서적 '지수 수명 모델' 공식)

[주의 - 정직하게 말해야 할 점]
  * 실제 크립토크롬(FAD-트립토판)은 원자핵 수십 개가 관여한다. 여기서는 원리를 보이기 위해 1개만 넣었다.
  * Azz = 1 mT, 수명 1 μs는 실제 크기와 '같은 자릿수'의 시연용 값이다. 실험값을 맞춘 것이 아니다.
  * 철새 부분(망막 패턴 -> 방향 선택)은 행동 모델이며, 실제 신경 처리 방식은 아직 밝혀지지 않았다.

실행:  python radical_pair_quantum_v2.py          (그래프 창 띄우기)
       python radical_pair_quantum_v2.py --save   (figures/ 폴더에 PNG 저장)
필요:  pip install numpy matplotlib
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. 물리 상수와 시연용 매개변수
# ---------------------------------------------------------------------------
GAMMA_E = 1.76e11          # 전자의 자기회전비 (rad/s per T)  -> 1 μT 에서 약 1.76e5 rad/s
A_ZZ_mT = 1.0              # 초미세 결합 크기 (mT, 시연용; 실제 FAD 질소핵 결합과 같은 자릿수)
K_REC = 1.0e6              # 라디칼 쌍이 반응해 사라지는 속도 (1/s) -> 평균 수명 1 μs
EARTH_FIELD_uT = 50.0      # 지구 자기장 세기 (위도에 따라 약 25~65 μT)

# 스핀-1/2 연산자 (파울리 행렬 / 2)
sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
I2 = np.eye(2, dtype=complex)


def kron3(a, b, c):
    """전자A ⊗ 전자B ⊗ 원자핵 순서로 연산자를 8x8로 확장."""
    return np.kron(np.kron(a, b), c)


# 각 입자의 스핀 연산자 (8x8)
S1 = [kron3(s, I2, I2) for s in (sx, sy, sz)]   # 전자 A
S2 = [kron3(I2, s, I2) for s in (sx, sy, sz)]   # 전자 B
In = [kron3(I2, I2, s) for s in (sx, sy, sz)]   # 원자핵

# 단일항 투영 연산자 P_S = 1/4 - S1·S2
P_S = 0.25 * np.eye(8) - sum(S1[i] @ S2[i] for i in range(3))
M_NUC = 2   # 원자핵 상태 수 (스핀-1/2 -> 2)


def field_vector(B_uT, theta_deg, phi_deg=0.0):
    """분자 좌표계에서 본 자기장 벡터 (T). theta = 분자 z축과 자기장 사이 각도."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return B_uT * 1e-6 * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def hamiltonian(B_uT, theta_deg, phi_deg=0.0):
    """H/ħ (rad/s). 제이만(두 전자) + 축 이방성 초미세(전자 A - 원자핵)."""
    Bvec = field_vector(B_uT, theta_deg, phi_deg)
    H = GAMMA_E * sum(Bvec[i] * (S1[i] + S2[i]) for i in range(3))
    H = H + GAMMA_E * A_ZZ_mT * 1e-3 * (S1[2] @ In[2])
    return H


def singlet_yield(B_uT, theta_deg, phi_deg=0.0, k=K_REC):
    """단일항 경로로 반응한 비율 Φ_S (0~1). 나머지 1-Φ_S 가 삼중항 경로."""
    w, V = np.linalg.eigh(hamiltonian(B_uT, theta_deg, phi_deg))
    Ps = V.conj().T @ P_S @ V                      # 고유상태 기준으로 바꾼 P_S
    dw = w[:, None] - w[None, :]                   # 에너지 차이 ω_mn
    return float(np.real(np.sum(np.abs(Ps) ** 2 * k**2 / (k**2 + dw**2))) / M_NUC)


def singlet_population(t, B_uT, theta_deg, k=K_REC):
    """시간에 따라 '아직 남아 있는' 라디칼 쌍 중 단일항인 양 (수명 감소 포함)."""
    w, V = np.linalg.eigh(hamiltonian(B_uT, theta_deg))
    Ps = V.conj().T @ P_S @ V
    dw = w[:, None] - w[None, :]
    amp = np.abs(Ps) ** 2 / M_NUC
    return np.array([np.real(np.sum(amp * np.cos(dw * ti))) for ti in t]) * np.exp(-k * t)


def orientation_averaged_yield(B_uT, n=24):
    """분자가 무작위 방향(용액 속)일 때의 평균 Φ_S. 세기에 대한 반응을 보려고 사용."""
    th = np.linspace(0, 90, n)
    w = np.sin(np.radians(th))
    return float(np.sum([singlet_yield(B_uT, x) for x in th] * w) / np.sum(w))


# ---------------------------------------------------------------------------
# 1. 철새 모델: 망막 수용체 -> 신호 패턴 -> 방향 선택 (세 단계를 분리)
# ---------------------------------------------------------------------------
INCLINATION_DEG = 53.0      # 복각: 자기력선이 지면과 이루는 각 (서울 부근 약 53°)

# 각도(0~90°)별 Φ_S 를 미리 계산해 두고 보간 -> 빠르게 사용
_TH_TABLE = np.linspace(0, 90, 91)
_PHI_TABLE = None


def _phi_table(B_uT):
    global _PHI_TABLE
    if _PHI_TABLE is None or _PHI_TABLE[0] != B_uT:
        _PHI_TABLE = (B_uT, np.array([singlet_yield(B_uT, t) for t in _TH_TABLE]))
    return _PHI_TABLE[1]


def earth_field(B_uT=EARTH_FIELD_uT, inclination=INCLINATION_DEG, flip_vertical=False):
    """지구 좌표 (x=동, y=북, z=위)에서의 자기장 벡터 (μT). 북반구: 북쪽+아래쪽을 향함."""
    inc = np.radians(inclination)
    vz = -np.sin(inc) * (-1 if flip_vertical else 1)
    return B_uT * np.array([0.0, np.cos(inc), vz])


def receptor_signal(heading_deg, Bvec_uT, rng, noise=0.03, contrast=1.0):
    """
    머리 방향(heading)에서, '앞-아래 45°'를 보는 망막 수용체 분자의 Φ_S 신호.
    분자 축과 자기장 사이 각도에 따라 신호가 달라진다(양자 계산 표 사용).
    noise    : 광자 수가 유한해서 생기는 측정 잡음
    contrast : 1보다 작으면 방향 정보가 약해짐 (예: 라디오파 잡음이 스핀 상태를 흐트러뜨린 경우의 '단순화된' 표현)
    """
    B = np.linalg.norm(Bvec_uT)
    h = np.radians(heading_deg)
    d = np.array([np.sin(h) * np.cos(np.radians(45)), np.cos(h) * np.cos(np.radians(45)), -np.sin(np.radians(45))])
    if B < 1e-9:                                  # 자기장이 없으면 방향 정보도 없음
        base = singlet_yield(0.0, 0.0)
        return base + rng.normal(0, noise)
    cos_a = abs(np.dot(d, Bvec_uT) / B)           # 축만 의미 있음(N극/S극 구분 못 함) -> abs
    ang = np.degrees(np.arccos(np.clip(cos_a, 0, 1)))
    table = _phi_table(B)
    mean = np.interp(90, _TH_TABLE, table)
    val = np.interp(ang, _TH_TABLE, table)
    return mean + contrast * (val - mean) + rng.normal(0, noise)


def choose_heading(Bvec_uT, rng, want="south", **kw):
    """머리를 한 바퀴 돌리며 신호를 비교 -> 자기력선이 땅으로 '가장 가파르게' 들어가는 쪽 = 극 방향(북).
    가을 이동(want='south')이면 그 반대쪽으로 비행."""
    heads = np.arange(0, 360, 10)
    sig = np.array([receptor_signal(h, Bvec_uT, rng, **kw) for h in heads])
    # 이 모델에서는 수용체가 자기력선과 나란할수록(θ 작을수록) Φ_S 가 크다 -> 최대값 쪽이 극 방향
    pole = heads[np.argmax(sig)]
    return (pole + 180) % 360 if want == "south" else pole


def fly(days=30, cond="normal", seed=1):
    rng = np.random.default_rng(seed)
    kw = {}
    if cond == "normal":
        B = earth_field()
    elif cond == "no_field":
        B = np.zeros(3)
    elif cond == "flip_vertical":       # Wiltschko(1972) 실험: 수직 성분만 뒤집기
        B = earth_field(flip_vertical=True)
    elif cond == "flip_polarity":       # 수평·수직 모두 뒤집기 = N/S 극 반전
        B = -earth_field()
    elif cond == "rf_noise":            # Engels 등(2014): 약한 전자기 잡음 -> 방향성 상실
        B = earth_field(); kw = dict(contrast=0.05)
    pos = np.zeros((days + 1, 2))
    for d in range(days):
        hd = choose_heading(B, rng, **kw) + rng.normal(0, 15)   # 바람 등으로 인한 오차
        step = 50.0                                               # 하루 비행거리 (km, 임의)
        pos[d + 1] = pos[d] + step * np.array([np.sin(np.radians(hd)), np.cos(np.radians(hd))])
    return pos


# ---------------------------------------------------------------------------
# 2. 그래프
# ---------------------------------------------------------------------------
def setup_font():
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        try:
            from matplotlib import font_manager
            font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.family"] = f
            break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False


def fig_time_evolution():
    t = np.linspace(0, 1e-6, 1500)
    fig, ax = plt.subplots(figsize=(7, 4))
    for th, c in [(0, "#c0392b"), (90, "#2471a3")]:
        ax.plot(t * 1e6, singlet_population(t, EARTH_FIELD_uT, th), color=c,
                label=f"자기장이 분자 축과 {th}°")
    ax.set_xlabel("시간 (μs)"); ax.set_ylabel("남아있는 단일항 라디칼 쌍")
    ax.set_title("단일항 ↔ 삼중항 진동 (50 μT, 양자 계산)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    return fig


def fig_angle():
    th = np.linspace(0, 180, 91)
    fig, ax = plt.subplots(figsize=(7, 4))
    for B, c in [(25, "#7fb3d5"), (50, "#2471a3"), (65, "#1b4f72")]:
        y = [singlet_yield(B, x) for x in th]
        ax.plot(th, y, color=c, label=f"{B} μT")
    ax.set_xlabel("분자 축과 자기장 사이 각도 θ (°)"); ax.set_ylabel("단일항 수득률 Φ_S")
    ax.set_title("같은 세기라도 '방향'이 바뀌면 생성물 비율이 바뀐다")
    ax.axvline(90, color="gray", lw=.8, ls=":")
    ax.legend(title="자기장 세기"); ax.grid(alpha=.25); fig.tight_layout()
    return fig


def fig_strength():
    Bs = np.concatenate([[0], np.logspace(0, np.log10(10000), 50)])
    y = np.array([orientation_averaged_yield(b, n=90) for b in Bs])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(Bs[1:], 1 - y[1:], color="#b9770e", lw=2, label="v2 양자 계산: 삼중항 수득률")
    ax.axhline(1 - y[0], color="#b9770e", ls=":", lw=1)
    ax.axvline(50, color="green", ls="--", lw=1)
    ax.text(55, (1 - y).min(), " 지구 자기장 50 μT", color="green")
    ax.text(2000, 1 - y[0] + 0.006, "0 μT 일 때 값", color="#b9770e", fontsize=9)
    ax.set_xlabel("자기장 세기 (μT, 로그 눈금)"); ax.set_ylabel("삼중항 경로 비율 (무작위 방향 평균)")
    ax.set_title("세기에 대한 반응은 '커질수록 증가'가 아니다\n(증가 → 평탄 → 감소, 모양은 분자마다 다름)")
    ax.grid(alpha=.25, which="both"); ax.legend(); fig.tight_layout()
    return fig


def fig_3d_sensitivity():
    """자기장 방향별 Φ_S 를 3D 곡면으로: 반지름 = 신호 크기(평균과의 차이를 확대)"""
    th = np.linspace(0, np.pi, 61); ph = np.linspace(0, 2 * np.pi, 61)
    TH, PH = np.meshgrid(th, ph)
    tab = np.array([singlet_yield(EARTH_FIELD_uT, np.degrees(x)) for x in th])
    Y = np.tile(tab, (len(ph), 1))
    r = 1 + 8 * (Y - Y.mean())
    X, Yc, Z = r * np.sin(TH) * np.cos(PH), r * np.sin(TH) * np.sin(PH), r * np.cos(TH)
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
    norm = (Y - Y.min()) / (Y.max() - Y.min())
    ax.plot_surface(X, Yc, Z, facecolors=plt.cm.coolwarm(norm), rstride=1, cstride=1, linewidth=0, alpha=.95, shade=False)
    ax.quiver(0, 0, -1.6, 0, 0, 3.2, color="k", arrow_length_ratio=.06)
    ax.text(0, 0, 1.75, "분자 축(z)", ha="center")
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=12, azim=30)
    ax.set_title("방향별 신호 3D 지도 (50 μT)\n빨강: Φ_S 큼 · 파랑: Φ_S 작음")
    fig.tight_layout()
    return fig


def fig_flights():
    conds = [("normal", "정상 지구 자기장", "#1e8449"), ("no_field", "자기장 없음", "#7f8c8d"),
             ("flip_vertical", "수직 성분만 반전", "#c0392b"), ("flip_polarity", "N/S 극 반전", "#2471a3"),
             ("rf_noise", "전자기 잡음(방향 대비 5%)", "#af7ac5")]
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, lab, col in conds:
        p = fly(cond=c, seed=3)
        if c == "normal":      # 극 반전과 경로가 완전히 같으므로 굵게 깔아 둔다
            ax.plot(p[:, 0], p[:, 1], "-", lw=7, alpha=.35, color=col, label=lab)
        elif c == "flip_polarity":
            ax.plot(p[:, 0], p[:, 1], "--o", ms=2.5, color=col, label=lab + " (정상과 같은 경로)")
        else:
            ax.plot(p[:, 0], p[:, 1], "-o", ms=2.5, color=col, label=lab)
    ax.plot(0, 0, "k*", ms=14)
    ax.set_xlabel("동쪽 방향 거리 (km)"); ax.set_ylabel("북쪽 방향 거리 (km)")
    ax.set_title("가을 이동(남쪽 목표) 30일 비행 시뮬레이션")
    ax.axis("equal"); ax.grid(alpha=.25); ax.legend(fontsize=9); fig.tight_layout()
    return fig


def main():
    setup_font()
    save = "--save" in sys.argv
    print("== 방향에 따른 단일항 수득률 (50 μT) ==")
    for th in (0, 30, 60, 90):
        print(f"  θ={th:>2}° : Φ_S = {singlet_yield(EARTH_FIELD_uT, th):.4f}")
    print("== 30일 뒤 위치 (x=동, y=북, km) ==")
    for c in ["normal", "no_field", "flip_vertical", "flip_polarity", "rf_noise"]:
        p = fly(cond=c, seed=3)[-1]
        print(f"  {c:<14}: x={p[0]:8.1f}, y={p[1]:8.1f}")
    figs = {"rp_time": fig_time_evolution(), "rp_angle": fig_angle(), "rp_strength": fig_strength(),
            "rp_3d": fig_3d_sensitivity(), "rp_flights": fig_flights()}
    if save:
        import os; os.makedirs("figures", exist_ok=True)
        for k, f in figs.items():
            f.savefig(f"figures/{k}.png", dpi=200)
        print("figures/ 폴더에 저장 완료")
    else:
        plt.show()


if __name__ == "__main__":
    main()
