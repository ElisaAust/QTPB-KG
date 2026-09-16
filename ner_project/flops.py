import torch
import os
from fvcore.nn import FlopCountAnalysis, parameter_count
from model.bert_bilstm_crf import BertBiLSTMCRF
from model.bbc import BertBiLSTMCRF_BBC

def check_model(model_class, name, **kwargs):
    """
    Calculate and print the FLOPs and parameter count of the specified model
    """
    # 1. Instantiate the model and set it to evaluation mode [cite: 1, 2]
    model = model_class(**kwargs).eval()
    
    # 2. Constructing dummy input (Batch=1, Seq_Len=128) [cite: 1, 2]
    input_ids = torch.randint(0, 21128, (8, 512))
    attention_mask = torch.ones_like(input_ids)
    
    # To avoid errors caused by the list returned by CRF decoding.
    # We pass in the dummy labels, have the model perform a forward pass, and return the loss tensor.
    labels = torch.randint(0, kwargs['num_tags'], (8, 512))
    inputs = (input_ids, attention_mask, labels) 

    print(f"正在分析 {name} 的复杂度...")
    
    with torch.no_grad():
        # 3. Calculate FLOPs (floating-point operations)
        flops = FlopCountAnalysis(model, inputs)
        # Suppress warnings about unsupported operators to make the output cleaner.
        flops.unsupported_ops_warnings(False) 
        total_flops = flops.total()
        
        # 4. Calculate Params (Total number of parameters)
        params = parameter_count(model)[""]
        
        print(f"[{name}] 结果报告:")
        print(f"  - 总参数量 (Params): {params / 1e6:.2f} M")
        print(f"  - 计算开销 (FLOPs) : {total_flops / 1e9:.2f} GFLOPs")
        print("-" * 40)

if __name__ == "__main__":
    # Configure this according to the actual path on your server
    BERT_DIR = "E:/PythonProject/ner_project/bert-base"
    
    print("🔍 模型效率对比分析 (FLOPs & Params)")
    print("=" * 45)

    # 1. Detection baseline model BBC [cite: 1, 2]
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

    # 2. Detection improvement model UFO [cite: 1, 2]
    try:
        check_model(
            BertBiLSTMCRF,
            "UFO (Improved)",
            bert_model_name=BERT_DIR,
            lstm_hidden=256,
            num_tags=31
        )
    except Exception as e:
        print(f"UFO 模型检测失败: {e}")
