"""
Training CSV Logger - 학습 결과를 CSV로 기록하는 모듈

LossResult: Loss 함수 반환 규격
TrainingLogger: CSV 기록 관리
CSVLoggingCallback: Trainer와 통합
"""
import os
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import torch
from transformers import TrainerCallback


@dataclass
class LossResult:
    """
    Loss 함수 반환 규격.
    모든 loss 함수는 이 형식을 반환해야 함.
    
    Attributes:
        total_loss: 역전파에 사용되는 최종 loss (torch.Tensor)
        components: 개별 loss 값들 {이름: 값} - CSV 컬럼명으로 사용
        outputs: 모델 출력 (logits 등), 선택적
    """
    total_loss: torch.Tensor
    components: Dict[str, float] = field(default_factory=dict)
    outputs: Any = None
    
    @property
    def component_names(self) -> List[str]:
        """CSV 컬럼명으로 사용할 loss component 이름들"""
        return list(self.components.keys())


class TrainingLogger:
    """
    학습 결과를 CSV로 기록.
    
    파일명 형식: {model_type}-{freeze_str}-{param_str}-{epochs}ep-{save_info}-{date_str}-{time_str}.csv
    저장 위치: result/
    """
    
    def __init__(
        self,
        model_type: str,
        freeze_str: str,
        param_str: str,
        epochs: int,
        save_info: str,
        output_dir: str = "result"
    ):
        """
        Args:
            model_type: 모델 타입 (e.g., "ministral_3_3b_instruct")
            freeze_str: freeze 설정 (e.g., "full", "layer_10")
            param_str: 파라미터 수 (e.g., "3.2B")
            epochs: 총 학습 epoch 수
            save_info: 저장 전략 정보 (e.g., "epoch", "500step")
            output_dir: CSV 저장 폴더 (default: "result")
        """
        # 파일명 생성
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        
        filename = f"{model_type}-{freeze_str}-{param_str}-{epochs}ep-{save_info}-{date_str}-{time_str}.csv"
        
        # result 폴더 생성
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, filename)
        
        self.records: List[Dict] = []
        
        print(f"[TrainingLogger] Will save to: {self.output_path}")
    
    @classmethod
    def from_args(cls, args, total_params: int) -> "TrainingLogger":
        """
        argparse args에서 직접 생성.
        
        Args:
            args: argparse로 파싱된 arguments
            total_params: 모델 총 파라미터 수
            
        Returns:
            TrainingLogger 인스턴스
        """
        freeze_str = args.freeze_until_layer if args.freeze_until_layer else "full"
        param_str = f"{total_params/1e9:.1f}B"
        
        if args.save_strategy == "steps":
            save_info = f"{args.save_steps}step"
        else:
            save_info = f"{args.save_strategy}"
        
        return cls(
            model_type=args.model_type,
            freeze_str=freeze_str,
            param_str=param_str,
            epochs=args.epochs,
            save_info=save_info
        )
    
    def log(
        self,
        step: int,
        epoch: float,
        loss_result: Optional[LossResult] = None,
        predict: Optional[str] = None,
        label: Optional[str] = None,
        **extra
    ):
        """
        한 스텝의 결과 기록.
        
        Args:
            step: Global step number
            epoch: Current epoch (float, e.g., 0.5, 1.0)
            loss_result: LossResult 객체 (loss components 포함)
            predict: 모델 예측 텍스트 (선택)
            label: 정답 텍스트 (선택)
            **extra: 추가 기록할 필드들
        """
        record = {
            "step": step,
            "epoch": epoch,
            "predict": predict,
            "label": label,
        }
        
        # Loss components 추가 (동적 컬럼)
        if loss_result is not None:
            total_loss_val = loss_result.total_loss.item() \
                if isinstance(loss_result.total_loss, torch.Tensor) \
                else loss_result.total_loss
            record["total_loss"] = total_loss_val
            record.update(loss_result.components)
        
        record.update(extra)
        self.records.append(record)
    
    def save(self):
        """CSV 파일로 저장 (덮어쓰기)"""
        if not self.records:
            print("[TrainingLogger] No records to save.")
            return
        
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"[TrainingLogger] Saved {len(self.records)} records to {self.output_path}")


class CSVLoggingCallback(TrainerCallback):
    """
    Trainer Callback으로 CSV 저장을 model checkpoint와 동기화.
    
    - on_save: checkpoint 저장 시 CSV도 저장
    - on_train_end: 학습 종료 시 최종 저장
    """
    
    def __init__(self, logger: TrainingLogger):
        """
        Args:
            logger: TrainingLogger 인스턴스
        """
        self.logger = logger
    
    def on_save(self, args, state, control, **kwargs):
        """모델 체크포인트 저장할 때 CSV도 저장"""
        self.logger.save()
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """학습 종료 시 최종 저장"""
        self.logger.save()
        return control
