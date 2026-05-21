# 语音指令识别系统

这是一个基于 PyQt5 + sherpa-onnx 的本地实时语音指令识别程序。

## 功能

程序识别固定格式的中文语音命令：

```text
wake_word + 几号机器人 + 方位 + 距离 + 任务
```

默认 wake_word：

```text
迈克同志
```

默认 stop_word：

```text
over
```

示例语音：

```text
迈克同志，三号机器人向东南前进五米执行侦查
```

输出格式：

```json
{
  "valid": true,
  "wake_word": "迈克同志",
  "object": "3号机器人",
  "direction": "东南",
  "distance_m": 5,
  "task": "侦查"
}
```

## 界面按钮

- `开启/关闭`：开启或关闭麦克风监听，开启后会实时识别 wake_word。
- `开始识别`：手动唤醒，不用说 wake_word，直接说命令。
- `停止识别`：结束当前命令识别并显示 JSON。
- 识别中说出 `stop_word`，默认 `over`，会自动结束当前命令识别并显示 JSON。
- 识别中 3 秒没有声音，会自动停止本次识别并输出 JSON。

## 模型文件

模型文件需要放在项目根目录：

```text
model.int8.onnx
tokens.txt
bbpe.model
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动

```bash
python main.py
```

## 识别范围

- 操作对象：一号到十号机器人，也兼容阿拉伯数字。
- 方位：东、南、西、北、东南、东北、西南、西北。
- 任务：侦查、打击、警卫。
