
# Chaos Lab
[![Build Status](https://dev.azure.com/zjysnow/ChaosLab/_apis/build/status/zjysnow.ChaosLab?branchName=master)](https://dev.azure.com/zjysnow/ChaosLab/_build/latest?definitionId=23&branchName=master)

# ChaosCV
目的是为了实现类似TinyCV的模块，以及方便测试等，同时能够用相同的接口封装多个深度学习的推理框架  
由于不会CMAKE，使用VS管理项目，不支持Linux环境  

目前包含的模块  
 - [x] LOG模块
 - [ ] Math模块
   - [x] 矩阵分解
   - [ ] 基础矩阵运算
- [ ] DNN模块
  - [x] BinaryOp
  - [x] Inner Product
- [x] 混淆矩阵

# Sandbox
不更新该文件夹，目的是为了方便在VS中测试代码而创建的沙盒项目  
请自行创建Sandbox项目进行测试