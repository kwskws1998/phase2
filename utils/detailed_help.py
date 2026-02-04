"""
Detailed Help System for Command Line Arguments

Usage: python script.py --arg_name --help
Example: python inference.py --refusal_threshold --help
"""

import sys

DETAILED_HELP = {
    # ===================
    # Common Arguments
    # ===================
    "model": """--model: 사용할 모델 지정

Usage:
  --model ministral_3_3b_instruct     기본 모델
  --model train/my_finetuned          학습된 모델 (model/train/ 에서 검색)

모델 경로 검색 순서:
  1. model/{model_name}/
  2. model/train/{model_name}/
""",

    "data_path": """--data_path: 데이터 파일 경로

Usage:
  --data_path sample.json             data/ 폴더 기준 상대 경로
  --data_path /absolute/path.json     절대 경로

데이터 형식:
  - JSON 파일
  - 각 항목에 "instruction" 또는 "text" 필드 필요
""",

    "max_length": """--max_length: 최대 생성 토큰 수

Usage:
  --max_length 512      짧은 응답
  --max_length 2048     중간 응답
  --max_length 32768    긴 응답 (기본값)

참고:
  - 모델의 max_position_embeddings (262144) 이내여야 함
  - 높을수록 메모리 사용량 증가
""",

    "temperature": """--temperature: 샘플링 온도

Usage:
  --temperature 0.3     보수적, 일관된 출력
  --temperature 0.7     균형 (기본값)
  --temperature 1.0     창의적
  --temperature 1.5+    매우 다양하지만 불안정

특성:
  - 낮을수록: 확률 높은 토큰 선호, 반복적
  - 높을수록: 다양하지만 오류 가능성 증가
""",

    "temperatures": """--temperatures: 다중 온도 설정 (RLHF용)

Usage:
  --temperatures 0.5,0.7,1.0,1.2      4개 응답 생성 (기본값)
  --temperatures 0.3,0.5              2개 응답 생성

각 온도에 대해 별도의 응답이 생성되어 비교 가능
""",

    # ===================
    # Refusal Mechanism
    # ===================
    "refusal_threshold": """--refusal_threshold: 불확실성 임계값

토큰 생성 시 logits의 표준편차가 이 값을 초과하면 거부하고 온도를 낮춤.

Usage:
  --refusal_threshold 1.5    엄격 (자주 거부)
  --refusal_threshold 3.0    기본값
  --refusal_threshold 5.0    관대 (거의 거부 안 함)

동작:
  uncertainty > threshold → 거부 → temp *= temp_decay → 재시도
""",

    "refusal_max_retries": """--refusal_max_retries: 토큰당 최대 재시도 횟수

Usage:
  --refusal_max_retries 3    기본값
  --refusal_max_retries 5    더 많은 재시도

재시도마다 온도가 낮아짐 (temp_decay 적용)
""",

    "refusal_temp_decay": """--refusal_temp_decay: 재시도 시 온도 감소율

Usage:
  --refusal_temp_decay 0.8    기본값 (20% 감소)
  --refusal_temp_decay 0.5    급격한 감소

예시 (temp=1.0, decay=0.8):
  retry 1: 1.0 × 0.8 = 0.8
  retry 2: 0.8 × 0.8 = 0.64
  retry 3: 0.64 × 0.8 = 0.512
""",

    "refusal_min_temp": """--refusal_min_temp: 최소 온도 하한

Usage:
  --refusal_min_temp 0.4    기본값
  --refusal_min_temp 0.1    더 낮은 하한 허용

온도가 이 값 아래로 내려가지 않음
""",

    "refusal_recovery_tokens": """--refusal_recovery_tokens: 온도 회복에 필요한 토큰 수

Usage:
  --refusal_recovery_tokens 3    기본값
  --refusal_recovery_tokens 5    더 천천히 회복

거부 후 원래 온도로 돌아가는 데 필요한 토큰 수
""",

    "refusal_recovery_method": """--refusal_recovery_method: 온도 회복 곡선

Available:
  linear        일정한 속도로 회복
  exponential   처음 빠르게, 나중에 느리게 (기본값)
  ease_out      처음 빠르게, 끝에서 부드럽게
  ease_in_out   처음과 끝이 부드럽게
  step          마지막 토큰에서 즉시 회복

Usage:
  --refusal_recovery_method exponential
""",

    "no_refusal": """--no_refusal: Refusal mechanism 비활성화

Usage:
  --no_refusal

이 옵션을 사용하면 불확실성 기반 거부 없이 일반 샘플링만 수행
""",

    # ===================
    # Training Arguments
    # ===================
    "loss_type": """--loss_type: 손실 함수 유형

Available:
  cross_entropy
    - 표준 Cross Entropy Loss
    - 가장 기본적인 언어 모델 학습

  heteroscedastic_cross_entropy
    - 불확실성 학습 포함 (Kendall & Gal, 2017)
    - 모델이 자신의 불확실성을 학습
    - 출력에 log_variance 포함

  gdpo
    - Grouped Direct Preference Optimization
    - 선호도 기반 학습
    - 여러 응답 생성 후 보상 기반 최적화

  heteroscedastic_gdpo
    - GDPO + 불확실성 학습
    - 선호도 최적화 + 불확실성 추정

Usage:
  --loss_type cross_entropy
  --loss_type heteroscedastic_gdpo
""",

    "epochs": """--epochs: 학습 에폭 수

Usage:
  --epochs 1    빠른 학습 (기본값)
  --epochs 3    일반적인 fine-tuning
  --epochs 10   작은 데이터셋

참고: 과적합 주의
""",

    "batch_size": """--batch_size: 배치 크기

Usage:
  --batch_size 1    메모리 절약
  --batch_size 2    기본값
  --batch_size 4    빠른 학습 (메모리 충분 시)

GPU 메모리에 따라 조절
""",

    "learning_rate": """--learning_rate: 학습률

Usage:
  --learning_rate 2e-5    기본값 (fine-tuning 권장)
  --learning_rate 1e-5    안정적
  --learning_rate 5e-5    빠른 학습

너무 높으면 발산, 너무 낮으면 학습 느림
""",

    "gdpo_group_size": """--gdpo_group_size: GDPO 그룹 크기

Usage:
  --gdpo_group_size 4    기본값

각 프롬프트에 대해 생성할 응답 수
그룹 내에서 보상 기반으로 선호도 학습
""",

    "gdpo_kl_coef": """--gdpo_kl_coef: KL 발산 패널티 계수

Usage:
  --gdpo_kl_coef 0.01    기본값
  --gdpo_kl_coef 0.1     강한 정규화

원본 모델에서 너무 벗어나지 않도록 제어
""",

    "heteroscedastic_T": """--heteroscedastic_T: Monte Carlo 샘플 수

Usage:
  --heteroscedastic_T 3    기본값 (메모리 효율적)
  --heteroscedastic_T 10   더 정확한 불확실성 추정

높을수록 정확하지만 메모리/시간 증가
""",

    "random_seed": """--random_seed: 랜덤 시드

Usage:
  --random_seed -1     무작위 (기본값)
  --random_seed 42     재현 가능한 결과

-1이면 매번 다른 결과, 양수면 동일 결과 재현
""",

    "debug": """--debug: 디버그 모드

Usage:
  --debug

상세 로깅 활성화:
  - 토큰별 생성 정보
  - 불확실성 값
  - 온도 변화
""",

    "top_k": """--top_k: Top-K 샘플링

Usage:
  --top_k 50    기본값
  --top_k 10    더 제한적
  --top_k 100   더 다양

확률 상위 K개 토큰에서만 샘플링
""",
}


def check_detailed_help():
    """
    Check for --arg_name --help pattern before argparse runs.
    Call this at the start of your script, before argparse.
    
    Usage:
        from utils.detailed_help import check_detailed_help
        check_detailed_help()
        
        # Then your normal argparse code
        parser = argparse.ArgumentParser(...)
    """
    if len(sys.argv) == 3 and sys.argv[2] == '--help' and sys.argv[1].startswith('--'):
        arg_name = sys.argv[1].lstrip('-').replace('-', '_')
        
        if arg_name in DETAILED_HELP:
            print(DETAILED_HELP[arg_name])
            sys.exit(0)
        else:
            print(f"No detailed help available for '--{arg_name.replace('_', '-')}'")
            print()
            print("Available detailed help topics:")
            for key in sorted(DETAILED_HELP.keys()):
                print(f"  --{key.replace('_', '-')}")
            sys.exit(1)


def get_available_help_topics():
    """Return list of arguments that have detailed help."""
    return sorted(DETAILED_HELP.keys())
