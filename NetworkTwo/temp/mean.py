def process_two_lines(text, process_func=lambda x, y: (x, y)):
    """
    处理多行字符串，每两行进行一次处理
    :param text: 输入的多行字符串
    :param process_func: 处理函数，接收两个参数（当前行，下一行），默认返回元组
    :return: 处理结果列表
    """
    lines = text.strip().split('\n')
    results = []
    
    for i in range(0, len(lines), 2):
        line1 = lines[i].strip()
        line2 = lines[i+1].strip() if i+1 < len(lines) else None
        results.append(process_func(line1, line2))
    
    return results

# 示例使用 ################################################

if __name__ == "__main__":
    # 示例输入
    sample_input = """Test [000200] 刀具_depth_dist: 0.000824
Test [000200] 刀具-遮挡_depth_dist: 0.001647
Test [000200] 手机_depth_dist: 0.001402
Test [000200] 手机-遮挡_depth_dist: 0.001336
Test [000200] 扳手_depth_dist: 0.000751
Test [000200] 扳手-遮挡_depth_dist: 0.001127
Test [000200] 易拉罐_depth_dist: 0.003860
Test [000200] 易拉罐-遮挡_depth_dist: 0.003798
Test [000200] 显示器_depth_dist: 0.004917
Test [000200] 显示器-遮挡_depth_dist: 0.005898
Test [000200] 水杯_depth_dist: 0.004734
Test [000200] 水杯-遮挡_depth_dist: 0.002841
Test [000200] 笔记本电脑_depth_dist: 0.008481
Test [000200] 笔记本电脑-遮挡_depth_dist: 0.006107
Test [000200] 键盘_depth_dist: 0.002797
Test [000200] 键盘-遮挡_depth_dist: 0.009633
Test [000200] 刀具_cd: 0.083134
Test [000200] 刀具-遮挡_cd: 0.094761
Test [000200] 手机_cd: 0.053339
Test [000200] 手机-遮挡_cd: 0.043655
Test [000200] 扳手_cd: 0.105092
Test [000200] 扳手-遮挡_cd: 0.104747
Test [000200] 易拉罐_cd: 0.018674
Test [000200] 易拉罐-遮挡_cd: 0.018558
Test [000200] 显示器_cd: 0.052375
Test [000200] 显示器-遮挡_cd: 0.054221
Test [000200] 水杯_cd: 0.035317
Test [000200] 水杯-遮挡_cd: 0.037993
Test [000200] 笔记本电脑_cd: 0.030229
Test [000200] 笔记本电脑-遮挡_cd: 0.031582
Test [000200] 键盘_cd: 0.054050
Test [000200] 键盘-遮挡_cd: 0.072850
Test [000200] 刀具_f1_tau: 0.518610
Test [000200] 刀具-遮挡_f1_tau: 0.340838
Test [000200] 手机_f1_tau: 0.563178
Test [000200] 手机-遮挡_f1_tau: 0.622054
Test [000200] 扳手_f1_tau: 0.495812
Test [000200] 扳手-遮挡_f1_tau: 0.452638
Test [000200] 易拉罐_f1_tau: 0.859173
Test [000200] 易拉罐-遮挡_f1_tau: 0.906545
Test [000200] 显示器_f1_tau: 0.587181
Test [000200] 显示器-遮挡_f1_tau: 0.552544
Test [000200] 水杯_f1_tau: 0.683285
Test [000200] 水杯-遮挡_f1_tau: 0.576533
Test [000200] 笔记本电脑_f1_tau: 0.722986
Test [000200] 笔记本电脑-遮挡_f1_tau: 0.720732
Test [000200] 键盘_f1_tau: 0.508454
Test [000200] 键盘-遮挡_f1_tau: 0.395661
Test [000200] 刀具_f1_2tau: 0.630614
Test [000200] 刀具-遮挡_f1_2tau: 0.571378
Test [000200] 手机_f1_2tau: 0.747478
Test [000200] 手机-遮挡_f1_2tau: 0.819666
Test [000200] 扳手_f1_2tau: 0.579182
Test [000200] 扳手-遮挡_f1_2tau: 0.575150
Test [000200] 易拉罐_f1_2tau: 0.964787
Test [000200] 易拉罐-遮挡_f1_2tau: 0.972363
Test [000200] 显示器_f1_2tau: 0.761811
Test [000200] 显示器-遮挡_f1_2tau: 0.756690
Test [000200] 水杯_f1_2tau: 0.858486
Test [000200] 水杯-遮挡_f1_2tau: 0.858633
Test [000200] 笔记本电脑_f1_2tau: 0.888701
Test [000200] 笔记本电脑-遮挡_f1_2tau: 0.865075
Test [000200] 键盘_f1_2tau: 0.772455
Test [000200] 键盘-遮挡_f1_2tau: 0.635193
"""

    # 示例处理方式1：简单合并
    def merge_lines(a, b):
        return f"{a} | {b}" if b else a

    # 示例处理方式2：计算行长度
    def calculate_length(a, b):
        len_a = len(a)
        len_b = len(b) if b else 0
        return len_a + len_b
    def calulate_mean(a:str, b:str):
        value1 = float(a.split(': ')[1])
        value2 = float(b.split(': ')[1])
        mean = ((value1+value2)/2) *100
        return a.split(': ')[0] +':' + str(mean)

    for result in process_two_lines(sample_input, calulate_mean):
        print(result)