# Adaptive-Weight

[English](README.md) | [한국어](README.ko.md)

긴 컨텍스트 LLM 서빙을 위한 점진적 가중치 정밀도: VRAM이 허용하는 동안 **int8**을 유지하고, 소프트 메모리 홀드 아래에서 레이어를 **int4**로 강등합니다. 처음부터 W4로 다시 로드하지 않습니다.

짧은 프롬프트는 W8 품질에 가깝게 유지됩니다. 컨텍스트가 커지면 occupancy 압력이 순위 순서대로 레이어를 강등하고, 강등이 소진되면 고정 AWQ int4와 같아집니다.

## 아이디어

| | Adaptive-Weight | 고정 AWQ int4 |
| --- | --- | --- |
| 로드 | 전체 W8, 압력이 있을 때만 강등 | 처음부터 전체 W4 |
| VRAM @ 2k–16k | **~9.4–9.5 GiB 평탄** (홀드 ≈ W8 footprint + 0.5 GiB) | KV와 함께 증가 (~6.2 → ~9.5 GiB) |
| 강등 소진 이후 (~16k) | W4+KV와 동일 (~10–11.4 GiB @ 18k–24k) | 동일 |
| 초기 `short_qa` @ 2k | hit | miss |
| IFStruct `pass_rate` (n=100) | **60%** | **58%** |

예측 occupancy가 소프트 타깃을 넘으면 강등이 발동합니다:

\[
\text{projected} = \frac{\texttt{alloc\_mib} + L \cdot \texttt{kv\_mib\_per\_tok}}{1024},\qquad
\text{target} = \texttt{W8\_alloc} + \texttt{hold\_headroom\_gib}
\]

\(\text{projected} > \text{target}\)이면 컨트롤러가 \(K\)개 레이어를 강등합니다. 여기서

\[
K = \mathrm{clamp}\!\left(\left\lceil\frac{(\text{projected}-\text{target})\cdot 1024}{\texttt{save\_mean\_mib}}\right\rceil,\; K_{\min},\; K_{\max}\right)
\]

레이어 순서와 레이어별 W8→W4 절감량은 오프라인 순위(`layer_rank.json`)에서 옵니다. 아래 수치에서는 KV 양자화를 끕니다.

스택: HuggingFace + Marlin packed GEMM; 모델 **Qwen3-8B**.

## 소프트 홀드 상수 (모델 / 디바이스마다 재튜닝)

이 값들은 **보편이 아닙니다**. 모델 크기, KV 레이아웃, 배치, GPU 메모리 클래스가 바뀌면 다시 측정하세요.

| 노브 | 역할 | 설정 방법 |
| --- | --- | --- |
| `kv_mib_per_tok` | 예측식에서 컨텍스트 토큰당 KV(+activation slack) MiB | 짧은 고정 W4 또는 Adaptive 세션에서 피팅: \(\Delta\texttt{alloc\_mib}/\Delta L\). 여기 기본값 ≈ **0.23** (Qwen3-8B, bs=1). |
| `hold_headroom_gib` | 콜드 W8 alloc 위의 소프트 천장 | **0.35–0.75** 근처에서 시작. 이 런에서는 **0.50**이 ≈ W8+0.5 GiB를 유지. 너무 타이트하면 조기 강등, 너무 느슨하면 발동 전 OOM. |
| `occ_k_min` / `occ_k_max` | 발동당 강등 레이어 수 | 작은 스텝 (예: **2–6**)은 홀드를 부드럽게 하고, 큰 \(K\)는 VRAM을 더 빨리 비우지만 stall이 커집니다. |
| `save_mean_mib` / `save_per_layer_mib` | \(K\)의 분모 | `build_awq_layer_rank.py`에서 (W8 vs W4 shard nbytes). 새 W8/W4 체크포인트 이후 다시 빌드. |
| `demote_order` | 어떤 레이어를 먼저 내릴지 | 같은 rank 빌드: BF16 대비 W4 재구성 오차가 낮은 레이어를 먼저 강등. 새 모델이면 항상 다시 빌드. |
| `budget_gib` | 하드 엔벨로프 (2-wave / 로깅) | 사용 가능한 디바이스 메모리에 맞춤. 소프트 경로는 주로 위의 `target`을 씁니다. |

전형적인 재튜닝 루프:

1. 대상 모델의 W8 / W4 AWQ (그리고 rank용 BF16)를 빌드합니다.
2. `build_awq_layer_rank.py` → 새 `layer_rank.json`.
3. 해당 디바이스에서 `kv_mib_per_tok`를 측정합니다 (KV를 미리 예약하면 `0`).
4. Adaptive VRAM이 강등 소진까지 W8+headroom 근처에서 평탄하고, 이후 W4+KV를 따라가도록 `hold_headroom_gib`를 고릅니다.
5. stall이나 overshoot가 너무 크면 `occ_k_*`를 스윕합니다.

CLI 플래그는 `hf_mixed_demote.py`에 있습니다 (`--kv-mib-per-tok`, `--hold-headroom-gib`, `--occ-k-min`, `--occ-k-max`, `--layer-rank`, `--budget-gib`).

## 결과 (L-sweep)

컨텍스트 길이 체크포인트: **2k…24k**, 2k 간격. 소프트 홀드: `kv_mib_per_tok≈0.23`, `hold_headroom_gib=0.50`.

### VRAM

![VRAM usage](docs/figs/20260806T052437Z/vram_scatter.png)

Adaptive-Weight는 **16k까지 ~10 GiB 이하**를 유지하고, 고정 int4는 컨텍스트와 함께 올라갑니다.

### 처리량

![tok/s](docs/figs/20260806T052437Z/toks_scatter.png)

같은 Marlin 경로입니다. Adaptive-Weight는 고정 int4와 비슷한 구간에 머무르고, mixed 레이어가 나타나면 작은 차이가 납니다.

### short_qa

![short_qa](docs/figs/20260806T052437Z/short_qa_scatter.png)

### 강등 stall (`swap_s`)

![swap](docs/figs/20260806T052437Z/swap_scatter.png)

라이브 W8→W4 레이어 morph는 W8 레이어가 남아 있는 동안 강등 스텝당 약 1.0–1.3 s가 들고, 모델이 전부 W4가 되면 0입니다.

### 핵심 수치

| L | Adaptive tok/s | AWQ int4 tok/s | Adaptive VRAM (MiB) | AWQ int4 VRAM (MiB) | Adaptive short_qa | AWQ short_qa | swap_s | 남은 W8 linear |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: |
| 2k | 19.0 | 19.9 | 9600 | 6316 | ✓ | ✗ | 0 | 252 |
| 4k | 16.6 | 14.5 | 9632 | 6801 | ✗ | ✗ | 1.05 | 217 |
| 8k | 11.3 | 13.3 | 9599 | 7776 | ✗ | ✗ | 1.29 | 140 |
| 12k | 9.3 | 10.5 | 9660 | 8748 | ✗ | ✗ | 1.07 | 70 |
| 16k | 7.8 | 8.5 | 9718 | 9721 | ✗ | ✗ | 1.07 | 0 |
| 18k | 7.1 | 7.7 | 10204 | 10206 | ✗ | ✗ | 0 | 0 |
| 24k | 5.5 | 5.9 | 11665 | 11667 | ✗ | ✗ | 0 | 0 |

플롯: `docs/figs/20260806T052437Z/`.

### 정확도 (IFStruct)

구조화 출력 `pass_rate`는 `qwen_ifstruct_eval.validate_response`로 측정합니다:

| Policy | n | pass_rate |
| --- | ---: | ---: |
| Adaptive-Weight (W8 start) | 100 | **0.60** (60/100) |
| AWQ int4 | 100 | **0.58** (58/100) |

## 재현

```bash
# 오프라인 강등 순서 (W8/W4/BF16 체크포인트가 있는 뒤)
python3 -u adaptive_weight/build_awq_layer_rank.py \
  --w4-dir quantized/Qwen3-8B-W4A16-AWQ \
  --w8-dir quantized_local/Qwen3-8B-W8A16-AWQ \
  --bf16-dir quantized/Qwen3-8B-BF16 \
  --out adaptive_weight/results/layer_rank.json

# locked clocks → L-sweep → scatter 4장
./adaptive_weight/run_beat_bench.sh \
  --out-dir results/$(date -u +%Y%m%dT%H%M%SZ)

# 플롯만
./adaptive_weight/run_beat_bench.sh --plot-only --run-dir results/<stamp>
```

IFStruct:

```bash
python3 -u adaptive_weight/hf_mixed_demote.py --mode ifstruct \
  --out-dir ifstruct_results/<stamp> \
  --dataset quantized/ifstruct_sample_100.jsonl \
  --ifstruct-policies "Adaptive-Weight,AWQ int4"
```

## 상태

- 동작 경로: HF + Marlin mixed demote (위 수치).
- 미결: 스톡 vLLM에서 mixed-precision hot-swap. 현재 데모는 HF Adaptive-Weight.
- 엣지 / unified-memory 타깃도 같은 소프트 홀드 아이디어를 쓰지만, 커널 지원이 그쪽 처리량을 아직 제한합니다.

코드 진입점: `adaptive_weight/hf_mixed_demote.py`, `occupancy_ctrl.py`, `inplace_w_replace.py`, `run_beat_bench.sh`. 플래그와 패키지 노트: [`adaptive_weight/README.md`](adaptive_weight/README.md).
