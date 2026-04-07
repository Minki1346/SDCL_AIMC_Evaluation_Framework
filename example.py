import torch
from torch import nn
from functools import partial
import numpy as np
import pandas as pd
import os
import math
from . import hw_effect_minki  # add analog PIM HW effect (2025.04.24, Chan-Gi Yook)
from typing import Optional

# FLOPS 측정을 위한 추가 import
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import contextmanager
import time
import psutil
import subprocess
import threading
import random
from pathlib import Path


forward_call_count = 0

# Weight quantization cache (conversion-time speedup)
USE_WEIGHT_CACHE = False
WEIGHT_CACHE_DIR = "./weight_cache"
WEIGHT_CACHE_TAG = "default"

# 니블 히스토그램 로그 파일 경로 (프로젝트 루트에 생성)
NIBBLE_HIST_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nibble_hist.csv"))

# set input, weight precision (2025.04.22, Chan-Gi Yook)
FP_PRECISION = 32  # user define parameter

if FP_PRECISION == 16:
    TARGET_DTYPE = torch.float16
elif FP_PRECISION == 32:
    TARGET_DTYPE = torch.float32
elif FP_PRECISION == 64:
    TARGET_DTYPE = torch.float64
else:
    raise ValueError(f"Unsupported FP_PRECISION: {FP_PRECISION}")

# ────────────────────────────────────────────────────────────
LAYER_HW_CONFIG = []




def save_tensor_to_csv(tensor, filename, column_offset=0, binary=False, transpose=False):
    """
    tensor를 CSV 또는 바이너리(.npy)로 저장.
    binary=True면 .npy로만 저장 (더 빠르고 용량 작음). filename의 확장자는 무시하고 .npy로 저장.
    transpose=True면 저장 전 행렬을 전치(T)함 (NeuroSIM activation 형식: in_features × seq_len).
    """
    # GPU 텐서인 경우 CPU로 이동
    if tensor.is_cuda:
        tensor = tensor.cpu()

    np_array = tensor.detach().numpy()

    # 3D 텐서인 경우 처리 (예: [1, n, m] 형태)
    if len(np_array.shape) == 3:
        np_array = np_array[0]  # 첫 번째 배치만 저장

    # transpose 옵션
    if transpose and np_array.ndim == 2:
        np_array = np_array.T

    n_cols = np_array.shape[1] if np_array.ndim > 1 else len(np_array)
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # 바이너리 저장: .npy로 저장 후 종료
    if binary:
        npy_path = os.path.splitext(filename)[0] + '.npy'
        if np.issubdtype(np_array.dtype, np.integer):
            arr_save = np_array.astype(np.uint8) if np_array.dtype == np.uint8 else np_array.astype(np.int32)
        elif np.issubdtype(np_array.dtype, np.floating):
            arr_flat = np_array.ravel()
            if np.all(np.isfinite(arr_flat)) and np.all(arr_flat == np.round(arr_flat)):
                # 0~255면 uint8, 그 외 정수면 int32
                mn, mx = arr_flat.min(), arr_flat.max()
                arr_save = np_array.astype(np.uint8) if mn >= 0 and mx <= 255 else np_array.astype(np.int32)
            else:
                arr_save = np_array.astype(np.float32)
        else:
            arr_save = np_array.astype(np.float32)
        np.save(npy_path, arr_save)
        return

    # 정수로 표현 가능하면 NumPy만 사용해 정수 포맷으로 저장 (pandas 대비 훨씬 빠름)
    if np.issubdtype(np_array.dtype, np.integer):
        arr_int = np_array.astype(np.int32)
        use_fast = True
    elif np.issubdtype(np_array.dtype, np.floating):
        arr_flat = np_array.ravel()
        if np.all(np.isfinite(arr_flat)) and np.all(arr_flat == np.round(arr_flat)):
            arr_int = np_array.astype(np.int32)
            use_fast = True
        else:
            arr_float = np_array.astype(np.float32)
            use_fast = False
    else:
        arr_float = np_array.astype(np.float32)
        use_fast = False

    if use_fast:
        with open(filename, 'w', newline='') as f:
            np.savetxt(f, arr_int, fmt='%d', delimiter=',')
    else:
        df = pd.DataFrame(arr_float)
        df.to_csv(filename, index=False, header=False)


def plot_simple_hist(tensor, title, filename, bins=256, plot_range=None):
    """
    Plots a histogram distribution of the given tensor values.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 텐서를 CPU로 옮기고 1차원으로 펼침
        data_np = tensor.detach().float().cpu().numpy().flatten()
        
        plt.figure(figsize=(10, 6))
        # plot_range가 주어지면 해당 범위 사용, 아니면 bins만 설정
        plt.hist(data_np, bins=bins, range=plot_range, log=True, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Count (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        plt.savefig(filename)
        plt.close()
        # print(f"Saved histogram to {filename}")
    except Exception as e:
        print(f"Failed to plot histogram for {title}: {e}")


def plot_distribution_hist(tensor, title, filename_prefix, z, s, k, xlim=None):
    """
    Plots a histogram distribution of the given tensor values.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 텐서를 CPU로 옮기고 1차원으로 펼침
        data_np = tensor.detach().cpu().numpy().flatten()
        
        plt.figure(figsize=(10, 6))
        plt.hist(data_np, bins=100, log=True) # 로그 스케일
        plt.title(f'{title} (z={z}, s={s}, k={k})')
        plt.xlabel('Value')
        plt.ylabel('Count (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        if xlim is not None:
            plt.xlim(xlim)
        else:
             # Set default xlim to -20 to 10 (scaled by 1e-6) if not provided, or use 1e-5 scale explicitly
             # User requested fixed 1e-5 scale visibility. 
             # Let's set a reasonable range that covers typical values but enforces the scale visual.
             pass # We will handle formatting below

        # Force x-axis to use scientific notation with 1e-5
        # The most robust way is to manually set the major formatter ticks
        
        ax = plt.gca()
        
        # Set format to scientific notation
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # Force the exponent to -5
        # We use a custom formatter wrapper or configure ScalarFormatter
        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-5, -5)) 
        ax.xaxis.set_major_formatter(formatter)
        
        # Explicitly setting the offset text can sometimes be overridden, 
        # but set_powerlimits((-5, -5)) tells matplotlib to use 1e-5 as the multiplier base 
        # if the values are within range.
        
        # If the values are very large (e.g. 1e-4), it might still prefer 1e-4.
        # To strictly force 1e-5 regardless of value magnitude (e.g. showing 10 x 10^-5 instead of 1 x 10^-4):
        class FixedOrderFormatter(matplotlib.ticker.ScalarFormatter):
            def __init__(self, order=0, useMathText=True):
                super().__init__(useMathText=useMathText)
                self.order = order
            def _set_order_of_magnitude(self):
                self.orderOfMagnitude = self.order
        
        # Force order of magnitude to -5
        fixed_formatter = FixedOrderFormatter(order=-5, useMathText=True)
        ax.xaxis.set_major_formatter(fixed_formatter)

        plt.tight_layout()
        filename = f'{filename_prefix}_{z}_{s}_{k}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Saved {title} plot to {filename}")
    except Exception as e:
        print(f"Failed to plot {title}: {e}")


def compute_nibble_histograms(uint8_like_tensor):
    """주어진 0~255 값의 텐서를 상위/하위 4비트로 분해하여 0~15 카운트를 반환."""
    int_tensor = uint8_like_tensor.to(torch.int32)
    high_nibble = (int_tensor >> 4) & 0xF
    low_nibble = int_tensor & 0xF
    high_counts = torch.bincount(high_nibble.flatten(), minlength=16)
    low_counts = torch.bincount(low_nibble.flatten(), minlength=16)
    return high_counts, low_counts


def append_nibble_hist_to_csv(block_idx, proj_name, forward_idx, w_high_cnt, w_low_cnt, x_high_cnt, x_low_cnt, csv_path: Optional[str] = None):
    """니블 히스토그램을 단일 CSV 파일에 누적 추가."""
    if csv_path is None:
        csv_path = NIBBLE_HIST_CSV

    # 디렉토리 보장
    dirname = os.path.dirname(csv_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # 행 데이터 구성
    row = {
        "forward_idx": int(forward_idx),
        "block_idx": int(block_idx) if block_idx is not None else -1,
        "proj_name": str(proj_name) if proj_name is not None else "",
    }

    for i, v in enumerate(w_high_cnt.tolist()):
        row[f"W_high_{i}"] = int(v)
    for i, v in enumerate(w_low_cnt.tolist()):
        row[f"W_low_{i}"] = int(v)
    for i, v in enumerate(x_high_cnt.tolist()):
        row[f"X_high_{i}"] = int(v)
    for i, v in enumerate(x_low_cnt.tolist()):
        row[f"X_low_{i}"] = int(v)

    # CSV에 append (헤더는 파일 없을 때만)
    file_exists = os.path.exists(csv_path)
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)


def visualize_conductance_mapping(cell_div, cell_div_V, start_idx, end_idx, z, s, k, detect, cellBit,
                                   max_rows=4095, max_cols=1022, show_actual_values=True, save_csv=True,
                                   stop_after=False):
    """Conductance mapping 시각화: 샘플 영역 잘라서 plot_conductance_mapping 호출 후 저장."""
    rows = max_rows
    cols = min(max_cols, cell_div.shape[1] - 1)
    cell_div_sample = cell_div[: rows + 1, start_idx : min(start_idx + cols + 1, end_idx)]
    cell_div_V_sample = cell_div_V[: rows + 1, start_idx : min(start_idx + cols + 1, end_idx)]
    output_filename = f"conductance_mapping_{z},{s},{k}_{detect}.png"
    hw_effect_minki.plot_conductance_mapping(
        cell_div_sample,
        cell_div_V_sample,
        detect,
        cellBit,
        output_filename,
        show_actual_values=show_actual_values,
        save_csv=save_csv,
    )
    if stop_after:
        raise ValueError("stop here")


# New, W,p-ch,unsigned quantization (2025.04.01, Minki Choi) 
@torch.no_grad()
def quantize_weight_per_channel_absmax(w, wl_weight=8):  # wl_weight 받아오도록 변경 (2025.04.06, Minki Choi)

    # FP_PRECISION에 따라 weight dtype 변환 (2025.04.22, Chan-Gi Yook)
    if TARGET_DTYPE is not None and w.dtype != TARGET_DTYPE:
        w = w.to(TARGET_DTYPE)

    w_min = w.min(dim=-1, keepdim=True)[0]
    w_max = w.max(dim=-1, keepdim=True)[0]

    q_max = 2 ** wl_weight - 1  # 255 for 8-bit
    scales = (w_max - w_min).clamp_(min=1e-5) / q_max
    zero_points = (-w_min / scales).round().clamp_(0, q_max)
    w_quant = w.clone()
    w_quant = (w_quant / scales + zero_points).round().clamp_(0, q_max)
    # print(f"w_quant:{w_quant}")
    # print(f"w_quant_min,max: {w_quant.min()},{w_quant.max()}")
    w_quant = (w_quant - zero_points) * scales
    # print(f"w_dequant_min,max: {w_dequant.min()},{w_dequant.max()}")

    return w_quant, scales, zero_points  # return w and scales and zeros (2025.04.07, Minki Choi)




# New, A,p-tok,unsigned quantization (2025.04.04, Minki Choi)
@torch.no_grad()
def quantize_activation_per_token_absmax(t, wl_activate=8):  # wl_activate 받아오도록 변경 (2025.04.06, Minki Choi)
    # print("\nactivation_per_token")
    # print(f"t_original:{t}")

    # FP_PRECISION에 따라 activation dtype 변환 (2025.04.22, Chan-Gi Yook)
    if TARGET_DTYPE is not None and t.dtype != TARGET_DTYPE:
        t = t.to(TARGET_DTYPE)

    t_shape = t.shape
    t.view(-1, t_shape[-1])

    t_min = t.min(dim=-1, keepdim=True)[0]
    t_max = t.max(dim=-1, keepdim=True)[0]

    q_max = 2 ** wl_activate - 1
    scales = (t_max - t_min).clamp_(min=1e-5) / q_max
    zero_points = (-t_min / scales).round().clamp_(0, q_max)
    t_quant = t.clone()
    t_quant = (t_quant / scales + zero_points).round().clamp_(0, q_max)

    # print(f"t_quant: {t_quant}")
    t_quant = (t_quant - zero_points) * scales

    # print(f"scales: {scales}")
    return t_quant, scales, zero_points  # return t and scales and zeros (2025.04.07, Minki Choi)



# Original W,p-ten,signed quantization
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, wl_weight=8):  # wl_weight 받아오도록 변경 (2025.04.06, Minki Choi)
    # w: (out_features, in_features)

    # FP_PRECISION에 따라 weight dtype 변환 (2025.04.22, Chan-Gi Yook)
    if TARGET_DTYPE is not None and w.dtype != TARGET_DTYPE:
        w = w.to(TARGET_DTYPE)

    scales = w.abs().max()
    q_max = 2 ** (wl_weight - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)

    #original code
    # w.div_(scales).round_().mul_(scales)

    #devide quantize and dequantize (2024.05.20, Chan-Gi Yook)
    w.div_(scales).round_()  # quantize된 가중치 tensor를 CIM device들에 mapping하기 위함 (2024.05.20, Chan-Gi Yook)
    w.mul_(scales)  # dequantize step (2024.05.20, Chan-Gi Yook)

    return w, scales  # return w and scales (2024.05.21, Chan-Gi Yook)




#Original A,p-ten,signed quantization
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, wl_activate=8):  # wl_activate 받아오도록 변경 (2025.04.06, Minki Choi)
    # print("\nactivation_per_tensor")
    # print(f"t_original:{t}")

    # FP_PRECISION에 따라 activation dtype 변환 (2025.04.22, Chan-Gi Yook)
    if TARGET_DTYPE is not None and t.dtype != TARGET_DTYPE:
        t = t.to(TARGET_DTYPE)

    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (wl_activate - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)


    #original code
    # t.div_(scales).round_().mul_(scales)

    #devide quantize and dequantize (2024.05.20, Chan-Gi Yook)
    t.div_(scales).round_()  # quantize된 activation tensor를 CIM device들에 input으로 넣기 위함 (2024.05.20, Chan-Gi Yook)
    # print(f"t_quant:{t}")
    t.mul_(scales)  # dequantize step (2024.05.20, Chan-Gi Yook)
    # print(f"t_dequant:{t}\n")
    return t, scales  # return t and scales (2024.05.22, Chan-Gi Yook)

# Transformer Layer & projection selection
PROJ_NAME_LIST = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
BLOCK_IDX_LIST = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]


def _get_weight_cache_files(block_idx, layer_idx, proj_name, weight_quant, wl_weight):
    """
    Weight quantization cache file paths.
    - weight_dequant.npy: dequantized weight tensor (float32 저장)
    - weight_scales.npy: per-channel/per-tensor scales tensor (float32 저장)
    - weight_zero_points.npy: zero points tensor (float32 저장)
    """
    if block_idx is None or layer_idx is None or proj_name is None:
        return None

    cache_dir = os.path.join(
        WEIGHT_CACHE_DIR,
        WEIGHT_CACHE_TAG,
        f"fp{FP_PRECISION}",
        f"wq={weight_quant}_wlw={wl_weight}",
        f"block={block_idx}",
        f"layer={layer_idx}",
        str(proj_name),
    )
    return {
        "dir": cache_dir,
        "weight": os.path.join(cache_dir, "weight_dequant.npy"),
        "scales": os.path.join(cache_dir, "weight_scales.npy"),
        "zero_points": os.path.join(cache_dir, "weight_zero_points.npy"),
    }


def _try_load_weight_cache(block_idx, layer_idx, proj_name, weight_quant, wl_weight):
    paths = _get_weight_cache_files(block_idx, layer_idx, proj_name, weight_quant, wl_weight)
    if paths is None:
        return None

    if not (os.path.exists(paths["weight"]) and os.path.exists(paths["scales"]) and os.path.exists(paths["zero_points"])):
        return None

    weight_dequant = torch.from_numpy(np.load(paths["weight"])).to(torch.bfloat16)
    weight_scales = torch.from_numpy(np.load(paths["scales"]))
    weight_zero_points = torch.from_numpy(np.load(paths["zero_points"])).to(torch.bfloat16)
    return weight_dequant, weight_scales, weight_zero_points


def _save_weight_cache(
    block_idx,
    layer_idx,
    proj_name,
    weight_quant,
    wl_weight,
    weight_dequant,
    weight_scales,
    weight_zero_points,
):
    paths = _get_weight_cache_files(block_idx, layer_idx, proj_name, weight_quant, wl_weight)
    if paths is None:
        return

    os.makedirs(paths["dir"], exist_ok=True)
    np.save(paths["weight"], weight_dequant.detach().cpu().float().numpy())
    np.save(paths["scales"], weight_scales.detach().cpu().float().numpy())
    np.save(paths["zero_points"], weight_zero_points.detach().cpu().float().numpy())


class W8A8Linear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            act_quant="per_token",
            quantize_output=False,

            # additional arguments
            proj_name: str = None,
            layer_idx: int = None,
            block_idx: int = None,

            # hardware parameters (2025.04.03, Minki Choi)
            wl_activate=8,
            wl_error=8,
            wl_weight=8,
            inference=1,
            cycle=10,
            cellBit=1,
            subArray=128,
            ADCprecision=5,
            vari=0,
            t=0,
            v=0,
            detect=0,
            target=0,
            model="llama",
            use_ir_drop=False,

    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.bfloat16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.bfloat16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        # save
        self.proj_name = proj_name
        self.layer_idx = layer_idx
        self.block_idx = block_idx

        # hardware parameters (2025.03.24, Minki Choi)
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_weight = wl_weight
        self.inference = inference
        self.cycle = cycle
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.model = model
        self.use_ir_drop = use_ir_drop

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = self.custom_quantize_activation_per_token  # (2025.04.03, Minki Choi)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = self.custom_quantize_activation_per_tensor  # (2024.05.22, Chan-Gi Yook)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant  # quantize_output 명령어가 주어지면, output_quant 방법도 act_quant 방법과 동일하게 정의
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    # (2024.05.22, Chan-Gi Yook)
    def custom_quantize_activation_per_tensor(self, t):
        """Custom method to call quantization and store scales."""

        # FP_PRECISION에 따라 activation dtype 변환 (2025.04.22, Chan-Gi Yook)
        if TARGET_DTYPE is not None and t.dtype != TARGET_DTYPE:
            t = t.to(TARGET_DTYPE)

        t, scales, zero_points = quantize_activation_per_tensor_absmax(t, self.wl_activate)
        # wl_activate 받아오도록 변경 (2025.04.06, Minki Choi)
        self.act_scales = scales  # Save scales for later use in forward or other methods
        self.act_zero_points = zero_points  # Save zero_points for later use in forward or other methods
        return t

    # (2025.04.03, Minki Choi)
    def custom_quantize_activation_per_token(self, t):
        """Custom method to call quantization and store scales."""

        # FP_PRECISION에 따라 activation dtype 변환 (2025.04.22, Chan-Gi Yook)
        if TARGET_DTYPE is not None and t.dtype != TARGET_DTYPE:
            t = t.to(TARGET_DTYPE)

        t, scales, zero_points = quantize_activation_per_token_absmax(t, self.wl_activate)
        # wl_activate 받아오도록 변경 (2025.04.06, Minki Choi)
        self.act_scales = scales  # Save scales for later use in forward or other methods
        self.act_zero_points = zero_points  # Save zero_points for later use in forward or other methods
        return t

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        global forward_call_count

        if TARGET_DTYPE is not None:
           # 캐시 로드 등으로 인해 self.weight/bias가 CPU에 남아있을 수 있어,
           # F.linear 전에 입력 x의 디바이스/타입으로 맞춰준다.
           x = x.to(dtype=TARGET_DTYPE)
           weight = self.weight.to(device=x.device, dtype=TARGET_DTYPE)
           bias = self.bias.to(device=x.device, dtype=TARGET_DTYPE) if self.bias is not None else None
        outputOriginal = torch.functional.F.linear(x, weight, bias)
        q_y = torch.zeros_like(outputOriginal)
        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_activate)

        cycle = self.cycle

        cellRange = 2 ** self.cellBit

        # numSubArray = int(weight.shape[1] / self.subArray)
        numSubArray = math.ceil(weight.shape[1] / self.subArray)
        q_x = self.act_quant(x)
        Sa = self.act_scales.to(dtype=TARGET_DTYPE).expand_as(q_x).contiguous()
        Sw = self.weight_scales.to(device=x.device, dtype=TARGET_DTYPE).expand_as(weight).contiguous()
        Za = self.act_zero_points.to(dtype=TARGET_DTYPE).expand_as(q_x).contiguous()
        Zw = self.weight_zero_points.to(device=x.device, dtype=TARGET_DTYPE).expand_as(weight).contiguous()
        quantized_weight = (weight / Sw + Zw).round().clamp(0, 2 ** self.wl_weight - 1)  # weight quantize
        quantized_x = (q_x / Sa + Za).round().clamp(0, 2 ** self.wl_activate - 1)  # activation quantize
        
        
        # 8비트 값을 상위/하위 4비트로 나눠 0~15 카운트 계산 (니블 히스토그램)
        # w_high_cnt, w_low_cnt = compute_nibble_histograms(quantized_weight)
        # x_high_cnt, x_low_cnt = compute_nibble_histograms(quantized_x)
        # append_nibble_hist_to_csv(self.block_idx, self.proj_name, forward_call_count, w_high_cnt, w_low_cnt, x_high_cnt, x_low_cnt)
        # gate/up/down: PyTorch weight (out,in)이 gate/up은 (11008,4096), down은 (4096,11008). NeuroSim NetWork CSV는 (out×in) 행렬로 4096×11008, 4096×11008, 11008×4096 기대 → 저장 시 전치하여 맞춤
        weight_transpose = self.proj_name in ("gate_proj", "up_proj", "down_proj")
        save_tensor_to_csv(quantized_weight, f"./weight_int8/{self.block_idx}/{self.proj_name}/weight_{self.block_idx},{self.proj_name}.csv", binary=False, transpose=weight_transpose)
        save_tensor_to_csv(quantized_x, f"./activation_int8/{self.block_idx}/{self.proj_name}/activation_{self.block_idx},{self.proj_name}.csv", binary=False, transpose=False)
    
        # MAC operation (2025.04.24, Minki Choi)
        if self.proj_name in PROJ_NAME_LIST and self.block_idx in BLOCK_IDX_LIST:
            # IMC operation (2025.04.24, Minki Choi)
            print(f"[IMC operation] block={self.block_idx} proj={self.proj_name} ADCprecision={self.ADCprecision} cellBit={self.cellBit}")
            output_orig = torch.functional.F.linear(x, weight, bias)
            output_of_z_sum = torch.zeros_like(output_orig)
            all_x_lsb = []

            # bit-slicing quantized activation
            for z in range(bitActivation):
                x_lsb = quantized_x % 2  # bit-slicing quantized activation
                all_x_lsb.append(x_lsb)
                quantized_x = torch.div(quantized_x - x_lsb, 2, rounding_mode='floor')
                output_of_s_sum = torch.zeros_like(output_of_z_sum)
                
                # subarray-wise weight tensor slicing (2025.04.28, Minki Choi)
                for s in range(numSubArray):
                    mask = torch.zeros_like(weight)
                    start_idx = s * self.subArray
                    end_idx = min((s + 1) * self.subArray, weight.shape[1])
                    mask[:,start_idx:end_idx] = 1
                    weight_div = quantized_weight * mask
                    # # CSV에는 start_idx~end_idx-1 열만 저장하고, 열 이름을 해당 인덱스로 맞춤
                    # weight_div_slice = weight_div[:, start_idx:end_idx]
                    # if z == 0:
                    #     save_tensor_to_csv(weight_div_slice, f"./weight_div/{self.block_idx}/{self.proj_name}/weight_div_{z},{s}.csv", column_offset=start_idx)


                    output_of_s_sumP = torch.zeros_like(output_of_s_sum)

                    # cellrange-wise weight element slicing (2025.04.28, Minki Choi)
                    for k in range(math.ceil(bitWeight / self.cellBit)):
                        cell_div = weight_div % cellRange  # bit-slicing element of quantized weight tensor (2025.04.28, Minki Choi)
                        weight_div = torch.div(weight_div - cell_div, cellRange, rounding_mode='floor')

                        # device layer selection for 3D memory device (detect 3만 레이어 구분, 나머지는 0)
                        if self.proj_name == "down_proj" and self.block_idx in {1} and k == 1 and self.detect == 3:
                            devicelayer = 1
                        elif self.detect == 3:
                            devicelayer = random.randint(1, 4)
                        else:
                            devicelayer = 0

                        # conductance mapping & D2D variation & endurance characteristics (vari: 0=노이즈 미적용, 1=sd_mapping 노이즈 적용)
                        cell_div_V,Vread,Ioff,Ion = hw_effect_minki.apply_noise(cell_div,self.detect,self.cellBit,cellRange,cycle,devicelayer,vari=self.vari)  
                        # seperate MSBs, LSBs resolution
                        # if k == 1: #MSBs
                        #     cell_div_V,Vread,Ioff,Ion = hw_effect_minki.apply_noise(cell_div,self.detect,self.cellBit-4,2 ** (self.cellBit-4),cycle,devicelayer)
                        # else: #LSBs
                        #     cell_div_V,Vread,Ioff,Ion = hw_effect_minki.apply_noise(cell_div,self.detect,self.cellBit,cellRange,cycle,devicelayer)
                        
                        # conductance mapping visualization
                        viz = False
                        if viz:
                            visualize_conductance_mapping(
                                cell_div, cell_div_V, start_idx, end_idx,
                                z, s, k, self.detect, self.cellBit,
                                stop_after=True,
                            )

                        # retention characteristics
                        cell_div_V, retention_ratio = hw_effect_minki.Retention(cell_div_V,self.t,self.v,self.detect,self.target,300,devicelayer)

                        # matrix multiplication results
                        output_of_k_sum = torch.functional.F.linear(x_lsb*Vread, cell_div_V*mask, bias) 
                        output_of_dummy_sum = torch.functional.F.linear(x_lsb, ((mask*(Ioff))), bias)
                        ideal_output = output_of_k_sum-output_of_dummy_sum
                        
                        
                        # IR drop (error map from ir_drop.csv, on/off by use_ir_drop)
                        if self.use_ir_drop:
                            output_of_k_sum, output_of_dummy_sum = hw_effect_minki.apply_ir_drop(
                                x_lsb, start_idx, end_idx,
                                output_of_k_sum, output_of_dummy_sum,
                                self.detect,
                            )

                        # ADC quantize & digitize
                        # uniform ADC resolution
                        ADC_output_new, adc_delta_new = hw_effect_minki.ADC_quantizeloss(output_of_k_sum, Ion, Ioff, self.subArray, self.ADCprecision)
                        ADC_output_dummy, _ = hw_effect_minki.ADC_quantizeloss(output_of_dummy_sum, Ion, Ioff, self.subArray, self.ADCprecision)
                        ADC_output_digitized = hw_effect_minki.ADC_digitize(ADC_output_new, ADC_output_dummy, Ion, Ioff, self.cellBit, retention_ratio)

                        # seperate MSBs, LSBs resolution
                        # if k == 1: #MSBs
                        #     ADC_output_new, adc_delta_new = hw_effect_minki.ADC_quantizeloss(output_of_k_sum, Ion, Ioff, self.subArray, self.ADCprecision)
                        #     ADC_output_dummy, _ = hw_effect_minki.ADC_quantizeloss(output_of_dummy_sum, Ion, Ioff, self.subArray, self.ADCprecision)
                        #     ADC_output_new_correction = hw_effect_minki.ADC_digitize(ADC_output_new, ADC_output_dummy, Ion, Ioff, self.cellBit-4, retention_ratio)
                        # else: #LSBs
                        #     ADC_output_new, adc_delta_new = hw_effect_minki.ADC_quantizeloss(output_of_k_sum, Ion, Ioff, self.subArray, self.ADCprecision-2)
                        #     ADC_output_dummy, _ = hw_effect_minki.ADC_quantizeloss(output_of_dummy_sum, Ion, Ioff, self.subArray, self.ADCprecision-2)
                        #     ADC_output_new_correction = hw_effect_minki.ADC_digitize(ADC_output_new, ADC_output_dummy, Ion, Ioff, self.cellBit, retention_ratio)
                        
                        # register                   
                        output_of_k_sumC_new = ADC_output_digitized

                        # shift & add
                        output_of_s_sumP.add_(output_of_k_sumC_new * (cellRange ** k))
             
                    output_of_s_sumP = output_of_s_sumP
                    output_of_s_sum.add_(output_of_s_sumP)

                    del output_of_s_sumP
                    del weight_div
                    del mask

                output_of_z_sum.add_(output_of_s_sum * (2 ** z))

                del output_of_s_sum

            output_of_z_sum = output_of_z_sum

            ########### normalized_y => y 로 복원 (2025.04.24, Minki Choi) ###########
            Sa_v = self.act_scales.to(dtype=TARGET_DTYPE)
            Sw_v = self.weight_scales.to(device=x.device, dtype=TARGET_DTYPE)
            Za_v = self.act_zero_points.to(dtype=TARGET_DTYPE)
            Zw_v = self.weight_zero_points.to(device=x.device, dtype=TARGET_DTYPE)


            #########a_new#########
            q_x_row_sum = q_x.sum(dim=2, keepdim=True) # confirmed
            a_left = q_x_row_sum / Sa_v # confirmed
            a_new = torch.matmul(a_left, Zw_v.transpose(0, 1))
        

            #########b_new#########
            weight_row_sum = weight.sum(dim=1, keepdim=True)
            b_right = weight_row_sum / Sw_v
            b_new = torch.matmul(Za_v, b_right.transpose(0, 1))

   
            #########c_new#########        
            if self.proj_name == "down_proj":
                c_new = torch.matmul(Za_v, Zw_v.transpose(0, 1)).mul_(11008)
            else:
                c_new = torch.matmul(Za_v, Zw_v.transpose(0, 1)).mul_(4096)

    
        
            # dequantization
            # Sa_v = self.act_scales.to(TARGET_DTYPE)
            # Sw_v = self.weight_scales.to(TARGET_DTYPE)
            # Za_v = self.act_zero_points.to(TARGET_DTYPE)
            # Zw_v = self.weight_zero_points.to(TARGET_DTYPE)

            # # ---- FLOPs helper (shape-based; broadcast/matmul을 실제 텐서 shape로 계산) ----
            # def _numel(t):
            #     return int(t.numel())

            # def _flops_elemwise(t_out):
            #     # div/mul/add 등 elementwise: output element 수만큼 1 FLOP로 근사
            #     return _numel(t_out)

            # def _flops_sum(t_in, dim: int):
            #     # torch.sum: 입력 원소 수만큼(예전 코드와 동일한 근사)으로 계산
            #     # (정확히는 (n-1) add 이지만, 데모/비교 목적상 입력 원소 수로 둠)
            #     return _numel(t_in)

            # def _flops_matmul(a, b):
            #     # a: (..., m, k), b: (..., k, n) 또는 (k, n)
            #     # broadcast batch dims는 a/b 중 큰 쪽으로 근사
            #     a_shape = list(a.shape)
            #     b_shape = list(b.shape)
            #     if len(a_shape) < 2 or len(b_shape) < 2:
            #         return 0
            #     m, k = a_shape[-2], a_shape[-1]
            #     n = b_shape[-1]
            #     batch_a = int(np.prod(a_shape[:-2])) if len(a_shape) > 2 else 1
            #     batch_b = int(np.prod(b_shape[:-2])) if len(b_shape) > 2 else 1
            #     batch = max(batch_a, batch_b)
            #     return int(2 * batch * m * k * n)

            # def _flops_linear(x_in, w):
            #     # x_in: (..., in_features), w: (out_features, in_features)
            #     x_shape = list(x_in.shape)
            #     if len(x_shape) < 1:
            #         return 0
            #     in_f = x_shape[-1]
            #     out_f = int(w.shape[0])
            #     batch = int(np.prod(x_shape[:-1])) if len(x_shape) > 1 else 1
            #     return int(2 * batch * in_f * out_f)

            # #1-1
            # q_x_row_sum = q_x.sum(dim=2, keepdim=True)
            # a_left = q_x_row_sum / Sa_v
            # a_new = torch.matmul(a_left, Zw_v.transpose(0, 1))
            # flops_1_1 = _flops_sum(q_x, dim=2) + _flops_elemwise(a_left) + _flops_matmul(a_left, Zw_v.transpose(0, 1))

            # #1-2
            # a_orig1 = q_x /Sa
            # a_orig2 = torch.functional.F.linear(a_orig1, Zw, bias)
            # flops_1_2 = _flops_elemwise(a_orig1) + _flops_linear(a_orig1, Zw)

            # #2-1
            # weight_row_sum = weight.sum(dim=1, keepdim=True)
            # b_right = weight_row_sum / Sw_v
            # # matmul: Za_v (1,I) @ b_right.T (I,O) -> (1,O) 이면 FLOPs 2*I*O; (B,S,I)@(I,O)면 2*B*S*I*O
            # b_new = torch.matmul(Za_v, b_right.transpose(0, 1))
            # flops_2_1 = _flops_sum(weight, dim=1) + _flops_elemwise(b_right) + _flops_matmul(Za_v, b_right.transpose(0, 1))

            # #2-2
            # b_orig1 = weight /Sw
            # b_orig2 = torch.functional.F.linear(Za, b_orig1, bias)
            # flops_2_2 = _flops_elemwise(b_orig1) + _flops_linear(Za, b_orig1)

            # #3-1
            # if self.proj_name == "down_proj":
            #     c_new = torch.matmul(Za_v, Zw_v.transpose(0, 1)).mul_(11008)
            # else:
            #     c_new = torch.matmul(Za_v, Zw_v.transpose(0, 1)).mul_(4096)
            # # mul_는 output 원소 수만큼 1 FLOP로 근사
            # flops_3_1 = _flops_matmul(Za_v, Zw_v.transpose(0, 1)) + _flops_elemwise(c_new)

            # #3-2
            # c_orig1 = torch.functional.F.linear(Za, Zw, bias)
            # flops_3_2 = _flops_linear(Za, Zw)

            # # FLOPs 프린트 (전체 레이어 출력 시 많으므로, 특정 block만 보려면 if self.block_idx==0: 등으로 제한 가능)
            # print(f"[FLOPs dequant] block={self.block_idx} proj={self.proj_name} q_x={tuple(q_x.shape)} weight={tuple(weight.shape)}")
            # print(f"  #1-1 (q_x_row_sum, a_left, a_new): {flops_1_1:,}")
            # print(f"  #1-2 (a_orig1, a_orig2):          {flops_1_2:,}")
            # print(f"  #2-1 (weight_row_sum, b_right, b_new): {flops_2_1:,}")
            # print(f"  #2-2 (b_orig1, b_orig2):          {flops_2_2:,}")
            # print(f"  #3-1 (c_new):                     {flops_3_1:,}")
            # print(f"  #3-2 (c_orig1):                   {flops_3_2:,}")
            # print(f"  total:                            {flops_1_1 + flops_1_2 + flops_2_1 + flops_2_2 + flops_3_1 + flops_3_2:,}")
            # if self.proj_name == "down_proj":
            #     raise ValueError("stop")


            subtract_c = output_of_z_sum - c_new
            subtract_b = subtract_c - b_new
            subtract_a = subtract_b - a_new
            SwT_expanded = torch.ones_like(output_of_z_sum)
            SwT = (
                self.weight_scales.to(device=x.device, dtype=TARGET_DTYPE)
                .transpose(0, 1)
                .unsqueeze(0)
            )
            SwT_expanded = (SwT_expanded * SwT)

            subtract_a_scaled = (
                subtract_a * SwT_expanded * self.act_scales.to(dtype=TARGET_DTYPE, device=x.device)
            )

        else:
            # GPU operation
            print(f"[GPU operation] block={self.block_idx} proj={self.proj_name}")
            # data type matching (BFloat16 and Float type mismatch resolution)
            
            if q_x.dtype != weight.dtype:
                q_x_same_dtype = q_x.to(weight.dtype)
            else:
                q_x_same_dtype = q_x
            
            subtract_a_scaled = torch.functional.F.linear(q_x_same_dtype, weight, bias)

        # output dequantization / requantization
        q_y = self.output_quant(subtract_a_scaled)

        forward_call_count += 1

        # convert to fp16 and return
        return q_y.to(torch.bfloat16)

    @staticmethod
    def from_float(
            module,
            weight_quant="per_channel",
            act_quant="per_token",
            quantize_output=False,

            # additional arguments (2025.04.23, Chan-Gi Yook)
            proj_name: str = None,
            layer_idx: int = None,
            block_idx: int = None,

            # hardware parameters (2025.04.03, Minki Choi)
            wl_activate=8,
            wl_error=8,
            wl_weight=8,
            inference=1,
            cycle=10,
            cellBit=1,
            subArray=128,
            ADCprecision=5,
            vari=0,
            t=0,
            v=0,
            detect=0,
            target=0,
            model="llama",
            use_ir_drop=False,

    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,

            # pass arguments (2025.04.23, Chan-Gi Yook)
            proj_name=proj_name,
            layer_idx=layer_idx,
            block_idx=block_idx,

            # hardware parameters (2025.04.03, Minki Choi)
            wl_activate=wl_activate,
            wl_error=wl_error,
            wl_weight=wl_weight,
            inference=inference,
            cycle=cycle,
            cellBit=cellBit,
            subArray=subArray,
            ADCprecision=ADCprecision,
            vari=vari,
            t=t,
            v=v,
            detect=detect,
            target=target,
            model=model,
            use_ir_drop=use_ir_drop,

        )
        use_weight_cache_local = (
            USE_WEIGHT_CACHE
            and block_idx is not None
            and layer_idx is not None
            and proj_name is not None
        )

        if use_weight_cache_local:
            cached = _try_load_weight_cache(block_idx, layer_idx, proj_name, weight_quant, wl_weight)
            if cached is not None:
                w_dequant, cached_scales, cached_zero_points = cached
                # weight_cache 로드는 CPU에서 시작하므로, 원본 module 텐서의 디바이스에 맞춰 이동
                target_device = module.weight.device
                new_module.weight = w_dequant.to(torch.bfloat16).to(target_device)
                new_module.weight_scales = cached_scales.to(target_device)
                new_module.weight_zero_points = cached_zero_points.to(target_device).to(torch.bfloat16)
                new_module.weight_quant_name = weight_quant
                if module.bias is not None:
                    new_module.bias = module.bias  # bias는 그대로 가져감
                print(
                    f"[LoadWeightCache] block={block_idx} "
                    f"layer={layer_idx} proj={proj_name} "
                    f"ADCprecision={ADCprecision} cellBit={cellBit}"
                )
                return new_module

        # block/layer-wise parameter print (2025.05.07, Chan-Gi Yook)
        print(
            f"[Quantize] block={block_idx} "
            f"layer={layer_idx} proj={proj_name} "
            f"ADCprecision={ADCprecision} cellBit={cellBit}"
        )

        # module.weight로 가져온 fp16 weight, scale
        if weight_quant == "per_channel":
            # weight dtype matching (2025.04.22, Chan-Gi Yook)
            if TARGET_DTYPE is not None and module.weight.dtype != TARGET_DTYPE:
                weight = module.weight.to(TARGET_DTYPE)
            w_dequant, new_module.weight_scales, new_module.weight_zero_points = quantize_weight_per_channel_absmax(
                # scales도 함께 저장 (2024.05.21, Chan-Gi Yook)
                # wl_weight 받아오도록 변경 (2025.04.06, Minki Choi)
                # zero_points 받아오도록 변경 (2025.04.07, Minki Choi)
                weight, wl_weight
            )
            # 결과를 다시 fp16으로 변환하여 저장
            new_module.weight = w_dequant.to(torch.bfloat16)
        elif weight_quant == "per_tensor":
            # FP_PRECISION에 따라 weight dtype 변환 (2025.04.22, Chan-Gi Yook)
            if TARGET_DTYPE is not None and module.weight.dtype != TARGET_DTYPE:
                weight = module.weight.to(TARGET_DTYPE)
            w_dequant, new_module.weight_scales = quantize_weight_per_tensor_absmax(
                # scales도 함께 저장 (2024.05.21, Chan-Gi Yook)
                # wl_weight 받아오도록 변경 (2025.04.06, Minki Choi)
                # zero_points 받아오도록 변경 (2025.04.07, Minki Choi)
                weight, wl_weight
            )
            # 결과를 다시 fp16으로 변환하여 저장
            new_module.weight = w_dequant.to(torch.bfloat16)
            # zero_points는 per_tensor에 없으므로 0으로 초기화
            new_module.weight_zero_points = torch.zeros_like(new_module.weight_scales).to(torch.bfloat16)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")

        new_module.weight_quant_name = weight_quant
        if use_weight_cache_local:
            _save_weight_cache(
                block_idx,
                layer_idx,
                proj_name,
                weight_quant,
                wl_weight,
                new_module.weight,
                new_module.weight_scales,
                new_module.weight_zero_points,
            )

        if module.bias is not None:
            new_module.bias = module.bias  # bias는 그대로 가져감
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"



def quantize_opt(
        model, weight_quant="per_tensor", act_quant="per_token", quantize_bmm_input=True
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.fc2 = W8A8Linear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, OPTAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_llama_like(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=False,

        # hardware parameters (2025.04.03, Minki Choi)
        wl_activate=8,
        wl_error=8,
        wl_weight=8,
        inference=1,
        cycle_L=10,
        cellBit_L=1,
        subArray_L=128,
        ADCprecision_L=5,
        vari_L=0,
        t_L=0,
        v_L=0,
        detect_L=0,
        target_L=0,
        use_ir_drop=False,
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    # (2025.04.28, Chan-Gi Yook)
    for block_idx, block in enumerate(model.model.layers):  # block index
        for layer_idx, (proj_name, parent, quant_out) in enumerate([  # layer index order 0~6
            ("q_proj", block.self_attn, True),
            ("k_proj", block.self_attn, True),
            ("v_proj", block.self_attn, True),
            ("o_proj", block.self_attn, False),
            ("gate_proj", block.mlp, False),
            ("up_proj", block.mlp, False),
            ("down_proj", block.mlp, False),
        ]):
            orig = getattr(parent, proj_name)
            # default value setting
            used_ADC = ADCprecision_L
            used_cell = cellBit_L

            # block/layer-wise override
            if block_idx < len(LAYER_HW_CONFIG) and layer_idx < len(LAYER_HW_CONFIG[block_idx]):
                cfg = LAYER_HW_CONFIG[block_idx][layer_idx]
                used_ADC = cfg.get("ADCprecision", used_ADC)
                used_cell = cfg.get("cellBit", used_cell)

            # use overridden values when calling from_float
            new = W8A8Linear.from_float(
                orig,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quant_out and quantize_bmm_input,

                proj_name=proj_name,
                block_idx=block_idx,
                layer_idx=layer_idx,

                # hardware parameters (2025.04.03, Minki Choi)
                wl_activate=wl_activate,
                wl_error=wl_error,
                wl_weight=wl_weight,
                inference=inference,
                cycle=cycle_L,
                cellBit=used_cell,  # parser 값 대신 used_cell (2025.04.28, Chan-Gi Yook)
                subArray=subArray_L,
                ADCprecision=used_ADC,  # parser 값 대신 used_ADC (2025.04.28, Chan-Gi Yook)
                vari=vari_L,
                t=t_L,
                v=v_L,
                detect=detect_L,
                target=target_L,
                use_ir_drop=use_ir_drop,
            )
            setattr(parent, proj_name, new)
    return model


def quantize_mixtral(
        model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = W8A8Linear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w2 = W8A8Linear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w3 = W8A8Linear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W8A8Linear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant
            )

    return model


def quantize_falcon(
        model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W8A8Linear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant
            )
            m.dense_4h_to_h = W8A8Linear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W8A8Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.dense = W8A8Linear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_model(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=False,

        # hardware parameters (2025.04.03, Minki Choi)
        wl_activate=8,
        wl_error=8,
        wl_weight=8,
        inference=1,
        cycle_L=10,
        cellBit_L=1,
        subArray_L=128,
        ADCprecision_L=5,
        vari_L=0,
        t_L=0,
        v_L=0,
        detect_L=0,
        target_L=0,
        use_ir_drop=False,
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,

            # hardware parameters (2025.04.03, Minki Choi)
            wl_activate=wl_activate,
            wl_error=wl_error,
            wl_weight=wl_weight,
            inference=inference,
            cycle_L=cycle_L,
            cellBit_L=cellBit_L,
            subArray_L=subArray_L,
            ADCprecision_L=ADCprecision_L,
            vari_L=vari_L,
            t_L=t_L,
            v_L=v_L,
            detect_L=detect_L,
            target_L=target_L,
            use_ir_drop=use_ir_drop,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

