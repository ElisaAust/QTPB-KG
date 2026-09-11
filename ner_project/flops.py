import torch
import os
from fvcore.nn import FlopCountAnalysis, parameter_count
# 确保路径正确，导入你的模型类 [cite: 1, 2]
from model.bert_bilstm_crf import BertBiLSTMCRF
from model.bbc import BertBiLSTMCRF_BBC

def check_model(model_class, name, **kwargs):
    """
    计算并打印指定模型的 FLOPs 和参数量
    """
    # 1. 实例化模型并设置为评估模式 [cite: 1, 2]
    model = model_class(**kwargs).eval()
    
    # 2. 构造虚拟输入 (Batch=1, Seq_Len=128) [cite: 1, 2]
    input_ids = torch.randint(0, 21128, (8, 512))
    attention_mask = torch.ones_like(input_ids)
    
    # 🔴 关键修复：为了避开 CRF decode 返回列表导致的报错 
    # 我们传入虚拟标签，让模型运行前向传播并返回 Loss Tensor
    labels = torch.randint(0, kwargs['num_tags'], (8, 512))
    inputs = (input_ids, attention_mask, labels) 

    print(f"正在分析 {name} 的复杂度...")
    
    with torch.no_grad():
        # 3. 计算 FLOPs (浮点运算数) 
        flops = FlopCountAnalysis(model, inputs)
        # 屏蔽不支持算子的警告，使输出更整洁 
        flops.unsupported_ops_warnings(False) 
        total_flops = flops.total()
        
        # 4. 计算 Params (总参数量) 
        params = parameter_count(model)[""]
        
        print(f"[{name}] 结果报告:")
        print(f"  - 总参数量 (Params): {params / 1e6:.2f} M")
        print(f"  - 计算开销 (FLOPs) : {total_flops / 1e9:.2f} GFLOPs")
        print("-" * 40)

if __name__ == "__main__":
    # 根据你服务器的实际路径设置 
    BERT_DIR = "E:/PythonProject/ner_project/bert-base"
    
    print("🔍 模型效率对比分析 (FLOPs & Params)")
    print("=" * 45)

    # 1. 检测基准模型 BBC [cite: 1, 2]
    try:
        check_model(
            BertBiLSTMCRF_BBC,
            "BBC (Baseline)",
            bert_model_name=BERT_DIR,
            lstm_hidden=256, 
            num_tags=31  # 建议填入你实际的标签数
        )
    except Exception as e:
        print(f"BBC 模型检测失败: {e}")

    # 2. 检测改进模型 UFO [cite: 1, 2]
    try:
        check_model(
            BertBiLSTMCRF,
            "UFO (Improved)",
            bert_model_name=BERT_DIR,
            lstm_hidden=256,
            num_tags=31
            # 🔴 修复：已删除 attention_type="ufo" 参数，因为模型内部已硬编码 
        )
    except Exception as e:
        print(f"UFO 模型检测失败: {e}")