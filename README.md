# 问题链接
http://www.apmcm.org/detail/2487
# UWImage-Restoration
To restore the image UW<br>
本项目用于学习水下图像恢复技术，作者为中山大学学生，比赛性质，非商业。欢迎交流。<br>
This project is for learning underwater image restoration techniques.<br>
The author is a student from Sun Yat-sen University. <br>
It is for competition purposes and non-commercial. <br>
Communication is welcome.<br>
# 解决问题：
1.对图像进行统计分析。<br>
2.构建退化模型<br>
3.构建反演复原模型<br>
4.构建复杂环境下的模型<br>
5.比较单一与其他模型给建议<br>
代码结构  
push和clone用于克隆和推送至github
# 图像处理函数
最小二乘法去噪，未写<br>
求取深度图像：此处使用了一个近似处理，由于红光在水体中衰减大，故选取另外两个通道的亮度值作为对比。<br>
将像素点绿蓝光中较大的值与红光进行差分得到一个值，将该值作为图像该像素点的深度值，该值越大深度越大(存疑)<br>
https://www.jcad.cn/cn/article/pdf/preview/d61a88a6-b1d4-4d9d-be72-4509a07338ee.pdf<br>
此处深度估计十分粗略，唯追求快速计算方出此下策。<br>
存在物体本身颜色带来的干扰，例如8号图像中的白色探测器的表面会因三通道亮度极大导致深度很大<br>

# Q1:  
showraw.py用于统计展示源图的各项基本信息<br>
sort.py用于分类图片，并输出至result.csv<br>
judge.py负责几种异常情况判断<br>
未完成<br>
# Q2:  
Q2_foggy文件用于调用foggy文件在图像上产生模糊并储存，本代码使用cv2中的高斯模糊代码，可设置卷积核大小<br>
Q2_colorshift文件用于调用colorshift文件在图像上产生色偏，调用实际物理模型，分通道进行。<br>
Q2_lowlight文件用于调用lowlight文件使图像产生低光异常。<br>
# Q3
通过物理模型尝试修复图像，其中涉及过滤噪声、景深的计算、求衰减函数t(x)最终进行颜色补偿。<br>
对于偏色情况，首先求取图片的深度图像，本处通过假设蓝色衰减最小，借助蓝色作为参考色道。<br>
先通过背景光估计蓝色色道衰减系数，然后计算其他色道的衰减系数。<br>
Richard W. Gould, Robert A. Arnone, and Paul M. Martinolich, "Spectral dependence of the scattering coefficient in case 1 and case 2 waters," Appl. Opt. 38, 2377-2383 (1999)<br>

# Q4

