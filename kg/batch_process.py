#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理所有BIO格式鸟类文件，生成三元组
"""

import os
import glob

def parse_bio_file(file_path):
    """
    解析BIO格式文件，返回每个鸟类的实体信息
    """
    birds = []
    current_bird = {}
    current_entity = None
    current_entity_type = None
    current_entity_text = ""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # 空行表示一个鸟类结束
        if not line:
            if current_entity and current_entity_type:
                # 保存最后一个实体
                if current_entity_type not in current_bird:
                    current_bird[current_entity_type] = []
                current_bird[current_entity_type].append(current_entity_text.strip())
            
            if current_bird:
                birds.append(current_bird)
                current_bird = {}
            
            current_entity = None
            current_entity_type = None
            current_entity_text = ""
            continue
        
        parts = line.split()
        if len(parts) != 2:
            continue
        
        char, tag = parts
        
        if tag == 'O':
            # 非实体，保存当前实体
            if current_entity and current_entity_type:
                if current_entity_type not in current_bird:
                    current_bird[current_entity_type] = []
                current_bird[current_entity_type].append(current_entity_text.strip())
            current_entity = None
            current_entity_type = None
            current_entity_text = ""
        
        elif tag.startswith('B-'):
            # 新实体开始
            # 先保存上一个实体
            if current_entity and current_entity_type:
                if current_entity_type not in current_bird:
                    current_bird[current_entity_type] = []
                current_bird[current_entity_type].append(current_entity_text.strip())
            
            # 开始新实体
            current_entity_type = tag[2:]
            current_entity_text = char
            current_entity = True
        
        elif tag.startswith('I-'):
            # 实体继续
            entity_type = tag[2:]
            if current_entity and entity_type == current_entity_type:
                current_entity_text += char
            else:
                # 新的实体类型，保存上一个
                if current_entity and current_entity_type:
                    if current_entity_type not in current_bird:
                        current_bird[current_entity_type] = []
                    current_bird[current_entity_type].append(current_entity_text.strip())
                
                current_entity_type = entity_type
                current_entity_text = char
                current_entity = True
    
    # 保存最后一个鸟类（如果有）
    if current_entity and current_entity_type:
        if current_entity_type not in current_bird:
            current_bird[current_entity_type] = []
        current_bird[current_entity_type].append(current_entity_text.strip())
    
    if current_bird:
        birds.append(current_bird)
    
    return birds


def bird_to_triples(bird_data):
    """
    将单个鸟类的实体数据转换为三元组
    """
    triples = []
    
    # 获取主中文名（第一个中文名）
    chinese_names = bird_data.get('中文名', [])
    if not chinese_names:
        return triples
    
    main_name = chinese_names[0]
    
    # 1. 学名（拉丁名）
    if '拉丁名' in bird_data:
        for latin_name in bird_data['拉丁名']:
            triples.append((main_name, '学名为', latin_name))
    
    # 2. 英文名
    if '英文名' in bird_data:
        for english_name in bird_data['英文名']:
            triples.append((main_name, '英文名为', english_name))
    
    # 3. 别名（除第一个中文名外的其他中文名）
    if len(chinese_names) > 1:
        for alias in chinese_names[1:]:
            triples.append((main_name, '又称', alias))
    
    # 4. 科
    if '科' in bird_data:
        for ke in bird_data['科']:
            triples.append((main_name, '属于科', ke))
    
    # 5. 目
    if '目' in bird_data:
        for mu in bird_data['目']:
            triples.append((main_name, '属于目', mu))
    
    # 6. 形态特征
    if '特征' in bird_data:
        for feature in bird_data['特征']:
            triples.append((main_name, '形态特征', feature))
    
    # 7. 叫声
    if '叫声' in bird_data:
        for sound in bird_data['叫声']:
            triples.append((main_name, '叫声为', sound))
    
    # 8. 体长
    if '体长' in bird_data:
        for length in bird_data['体长']:
            triples.append((main_name, '体长范围', length))
    
    # 9. 栖息地
    if '栖息地' in bird_data:
        for habitat in bird_data['栖息地']:
            triples.append((main_name, '栖息于', habitat))
    
    # 10. 分布地
    if '分布地' in bird_data:
        for distribution in bird_data['分布地']:
            triples.append((main_name, '分布于', distribution))
    
    # 11. 海拔
    if '海拔' in bird_data:
        for altitude in bird_data['海拔']:
            triples.append((main_name, '生活海拔', altitude))
    
    # 12. 食物
    if '食物' in bird_data:
        for food in bird_data['食物']:
            triples.append((main_name, '食物有', food))
    
    # 13. 迁徙情况
    if '迁徙情况' in bird_data:
        for migration in bird_data['迁徙情况']:
            triples.append((main_name, '迁徙类型', migration))
    
    # 14. IUCN
    if 'IUCN' in bird_data:
        for iucn in bird_data['IUCN']:
            triples.append((main_name, 'IUCN评级为', iucn))
    
    # 15. 中国保护等级
    if '中国保护等级' in bird_data:
        for protection in bird_data['中国保护等级']:
            triples.append((main_name, '国家保护级别为', protection))
    
    return triples


def process_all_files(input_dir, output_file):
    """
    批量处理所有txt文件并合并输出
    """
    # 获取所有txt文件，排除之前的train/dev/test
    all_files = glob.glob(os.path.join(input_dir, '*.txt'))
    # 过滤掉train.txt, dev.txt, test.txt
    txt_files = [f for f in all_files if not os.path.basename(f) in ['train.txt', 'dev.txt', 'test.txt']]
    
    if not txt_files:
        print("没有找到需要处理的txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个文件待处理:")
    for f in txt_files:
        print(f"  - {os.path.basename(f)}")
    
    all_triples = []
    total_birds = 0
    
    for file_path in sorted(txt_files):
        filename = os.path.basename(file_path)
        print(f"\n正在处理: {filename}")
        
        try:
            birds = parse_bio_file(file_path)
            print(f"  解析出 {len(birds)} 个鸟类")
            
            for bird in birds:
                triples = bird_to_triples(bird)
                all_triples.extend(triples)
                total_birds += 1
            
            print(f"  生成 {len(triples)} 个三元组")
            
        except Exception as e:
            print(f"  处理出错: {e}")
            continue
    
    # 保存所有三元组
    print(f"\n正在保存三元组到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for triple in all_triples:
            subject, relation, obj = triple
            f.write(f"{subject}\t{relation}\t{obj}\n")
    
    print(f"\n处理完成！")
    print(f"总计: {total_birds} 个鸟类, {len(all_triples)} 个三元组")
    
    return total_birds, len(all_triples)


def main():
    """
    主函数
    """
    input_dir = '../清理后的数据'
    output_file = 'all_birds_triples.txt'
    
    print("="*60)
    print("批量处理BIO格式鸟类数据")
    print("="*60)
    
    process_all_files(input_dir, output_file)
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
