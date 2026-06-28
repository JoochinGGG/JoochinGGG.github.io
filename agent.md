---
layout: default
title: 认识agent
---

# 1. 什么是 Agent？

Agent 可以理解为一种能够围绕目标进行多步决策和行动的智能系统。我们可以对一句话定义为：Agent 是一个能感知环境，并根据感知结果采取行动的系统。Agent 不是大模型。 Agent 的本质是：观察环境，然后选择动作

其工作的workflow应该是

Environment → Observation → Agent → Action → Environment （这很像强化学习） 

# 2. 最基础的Agent类型

## 2.1 Simple Reflex Agent

它只根据当前观察做动作，不考虑历史，也不考虑未来。

适用场景

环境很简单、当前观察足够决定动作。

比如：

自动门；
简单报警器；
简单温控系统；
简单扫地机器人规则。

## 2.2 Model-Based Reflex Agent

它不只看当前观察，还维护一个内部状态。 因为很多时候，agent 不能直接看到完整环境。 比如扫地机器人现在只看到当前位置，但它需要记得：哪些地方已经扫过？哪里有障碍物？刚才从哪里来？

适用场景

环境不是完全可见，需要记忆。

比如：

扫地机器人地图；
游戏 NPC 记住玩家位置；
简单导航系统；
无人机维护局部拓扑状态。

## 2.3 Goal-Based Agent

Agent 有一个明确目标，然后选择能让自己接近目标的动作。  

常用的算法会有：BFS
DFS
Dijkstra
A*
Planning
Search

适用场景
机器人导航；
游戏 AI 寻路；
自动任务规划；
物流路径规划；
网页 agent 完成指定任务。

## 2.4 Utility-Based Agent

它会给不同结果打分，选择效用最高的动作。
action = argmax utility(action)
适用场景
自动驾驶；
股票交易策略；
资源分配；
调度系统；
网络路由；
推荐系统；
无人机任务规划。

## 2.5 Learning Agent

它不是永远按照固定规则行动，而是根据结果调整策略。 
尝试动作 → 得到反馈 → 更新策略 → 下次更好

## 2.6 LLM Agent 

用LLM来做决策

## 2.7 Multi-Agent System：多智能体系统

多个 agent 在同一个环境中交互。共同影响 Environment

# 3 LLM Agent 

一个完整的 LLM Agent 通常由这些部分构成：LLM Agent = LLM + Goal + State/Memory + Tools + Action Space + Observation + Controller + Feedback + Safety

# 3.1 LLM 

负责：

理解用户目标；
分析当前状态；
生成计划；
选择下一步动作；
解释工具返回结果；
决定什么时候结束任务。
相当于传统agent里的policy

# 3.2 Goal 

# 3.3 State / Context

State 表示 agent 当前知道什么。

在传统 agent 里，state 可能是坐标、速度、地图。
在 LLM Agent 里，state 通常是文本上下文。
