import os
import jetson.inference
import jetson.utils

# 1. 加载 SSD-Mobilenet-v2 目标检测模型
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# 2. 定义两张输入图片的路径与输出路径（请确保图片在指定目录下）
image_paths = [
    {
        "input": "/home/nvidia/jetson-inference/data/images/test1.jpg",
        "output": "/home/nvidia/jetson-inference/data/images/test/test1_result.jpg",
    },
    {
        "input": "/home/nvidia/jetson-inference/data/images/test2.jpg",
        "output": "/home/nvidia/jetson-inference/data/images/test/test2_result.jpg",
    },
]

# 3. 循环处理两张图片
for index, img_info in enumerate(image_paths, start=1):
    input_path = img_info["input"]
    output_path = img_info["output"]

    print(f"\n==================== 处理第 {index} 张图片 ====================")
    print(f"输入路径: {input_path}")

    # 加载图片
    img = jetson.utils.loadImage(input_path)
    if img is None:
        print(f"错误: 无法加载图片 {input_path}，请检查路径是否正确！")
        continue

    # 执行检测
    detections = net.Detect(img)

    # 打印检测结果统计
    print(f"检测到的目标数量: {len(detections)}")

    # 遍历每个检测到的目标并输出详细参数
    for i, det in enumerate(detections, start=1):
        print(f"\n  --- 目标 {i} 属性数据 ---")
        print(f"  ClassID:    {det.ClassID}")
        print(f"  Confidence: {det.Confidence:.4f}")
        print(f"  Left:       {det.Left:.2f}")
        print(f"  Top:        {det.Top:.2f}")
        print(f"  Right:      {det.Right:.2f}")
        print(f"  Bottom:     {det.Bottom:.2f}")
        print(f"  Width:      {det.Width:.2f}")
        print(f"  Height:     {det.Height:.2f}")
        print(f"  Area:       {det.Area:.2f}")
        print(f"  Center:     ({det.Center[0]:.2f}, {det.Center[1]:.2f})")

    # 保存标注框后的图像
    jetson.utils.saveImage(output_path, img)
    print(f"\n结果图片已保存至: {output_path}")

print("\n==================== 所有检测任务完成 ====================")

