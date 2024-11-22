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
# Q1:  
showraw.py用于统计展示源图的各项基本信息<br>
sort.py用于分类图片，并输出至result.csv<br>
judge.py负责几种异常情况判断<br>
# Q2:  
Q2_foggy文件用于调用foggy文件在图像上产生模糊并储存，本代码使用cv2中的高斯模糊代码，可设置卷积核大小<br>
Q2_colorshift文件用于调用colorshift文件在图像上产生色偏，调用实际物理模型，分通道进行。<br>
Q2_lowlight文件用于调用lowlight文件使图像产生低光异常。<br>
push和clone用于克隆和推送至github
# Q3
通过物理模型尝试修复图像，其中涉及过滤噪声、景深的计算、求衰减函数t(x)最终进行颜色补偿。

# Q4

