# 演示冒烟报告

- 时间:2026-11-12 13:43:42
- 演示库:`data/demo.db` + `data/demo_milvus.db`(SQLite + Milvus Lite)
- 服务:127.0.0.1:18099(Langfuse / MCP 关闭)
- 模型:deepseek-flash
- 结果:**7/8 PASS**

| 探针 | 结果 | 耗时 | 断言与关键事件 |
|---|---|---|---|
| P1 商品咨询(检索+闸+引用) | PASS | 22.2s | ✅ 路由=knowledge(实际 knowledge)<br>✅ retrieve_knowledge 节点进出<br>✅ 置信度闸放过(top_score=0.9968)<br>✅ 带引用(实际 10 条)<br>✅ 答复非空<br>✅ 无 error 事件<br>intent=商品咨询/knowledge conf=0.95 gate=过(0.9968) finish=stop |
| P2 查物流(多步工具) | PASS | 4.5s | ✅ 路由=business(实际 business)<br>✅ 调用 query_logistics(实际 ['query_order', 'query_logistics'])<br>✅ 工具全部成功<br>✅ 无 error 事件<br>intent=物流/business conf=0.98 tools=query_order,query_logistics finish=stop |
| P3 指代消解+退光子流程 | PASS | 14.2s | ✅ 指代被补全(实际 changed=True)<br>✅ 补全结果含 1003('订单1003的大容量双肩通勤包能退吗')<br>✅ 路由=refund(实际 refund)<br>✅ refund_subflow 节点进出<br>✅ 无 error 事件<br>intent=退款退货/refund conf=0.9 gate=拦(0.115) finish=abstain |
| P4 投诉(升级按钮) | PASS | 1.9s | ✅ 路由=complaint(实际 complaint)<br>✅ 带转人工按钮(实际 ['transfer_human', 'create_ticket'])<br>✅ 带建工单按钮(实际 ['transfer_human', 'create_ticket'])<br>✅ 投诉路不调工具<br>intent=投诉/complaint conf=0.9 finish=complaint |
| P5 闲聊(零工具) | PASS | 1.8s | ✅ 路由=chitchat(实际 chitchat)<br>✅ 零工具调用<br>✅ finish=chitchat(实际 chitchat)<br>intent=闲聊/chitchat conf=0.95 finish=chitchat |
| P6 拒答(闸拦截) | PASS | 14.2s | ✅ 出现过 gate 事件<br>✅ 闸拦下(top_score=0.0284)<br>✅ finish=abstain(实际 abstain)<br>intent=商品咨询/knowledge conf=0.9 gate=拦(0.0284) finish=abstain |
| P7 缺单号(订单选择器) | PASS | 2.0s | ✅ 3 笔候选订单(实际 3)<br>✅ finish=need_order(实际 need_order)<br>intent=退款退货/refund conf=0.85 finish=need_order |
| P8 写操作确认流(interrupt→resume→落库) | **FAIL** | 22.4s | ✅ 出现 interrupt 确认卡片<br>✅ 卡片带描述<br>✅ resume 无 error 事件<br>✅ 恢复后跑完(实际 stop)<br>❌ tickets 表落 1 行(实际 0)<br>卡片:售后 | 用户反馈订单1001（无线蓝牙降噪耳机×1，实付¥399.00，下单日2026-08-07）显示已签收但未收到货。系统查<br>intent=None/None tools=query_order,query_logistics finish=stop |

判定:全 PASS = 现在能演。任何 FAIL 先修再演示——见 docs/测试与演示方案.md 第 4 节门禁。
