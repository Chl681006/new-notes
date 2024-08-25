# XTuner微调个人小助手认知

在本节中，将一步步带领大家体验如何使用 XTuner 完成个人小助手的微调！

> 整个过程大概需要90分钟我们就可以得到一个自己的小助手。

先看看微调效果：

|      | 微调前                                                       | 微调后                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 输入 | 请介绍一下你自己                                             | 请介绍一下你自己                                             |
| 输出 | [![img](https://raw.githubusercontent.com/wux-labs/ImageHosting/main/XTuner/image-11.png)](https://raw.githubusercontent.com/wux-labs/ImageHosting/main/XTuner/image-11.png) | [![img](https://raw.githubusercontent.com/wux-labs/ImageHosting/main/XTuner/image-12.png)](https://raw.githubusercontent.com/wux-labs/ImageHosting/main/XTuner/image-12.png) |

详细闯关任务请访问[闯关任务](https://github.com/InternLM/Tutorial/blob/camp3/docs/L1/XTuner/task.md)，提交作业助教老师批改后将获得 100 算力点奖励！！！

## 1 微调前置基础



本节主要重点是带领大家实现个人小助手微调，如果想了解微调相关的基本概念，可以访问[XTuner微调前置基础](https://github.com/InternLM/Tutorial/blob/camp3/docs/L1/XTuner/xtuner_finetune_basic.md)。

## 2 准备工作



**环境安装**：我们想要用简单易上手的微调工具包 XTuner 来对模型进行微调的话，第一步是安装 XTuner ！安装基础的工具是一切的前提，只有安装了 XTuner 我们才能够去执行后续的操作。

**前期准备**：在完成 XTuner 的安装后，我们下一步就需要去明确我们自己的微调目标了。我们想要利用微调做一些什么事情呢，然后为了实现这个目标，我们需要准备相关的硬件资源和数据。

**启动微调**：在确定了自己的微调目标后，我们就可以在 XTuner 的配置库中找到合适的配置文件并进行对应的修改。修改完成后即可一键启动训练！训练好的模型也可以仅仅通过在终端输入一行命令来完成转换和部署工作！

### 2.1 开发机准备



我们需要前往 [InternStudio](https://studio.intern-ai.org.cn/) 中创建一台开发机进行使用。

步骤1：登录InternStudio后，在控制台点击 “创建开发机” 按钮可以进入到开发机的创建界面。

步骤2：在 “创建开发机” 界面，选择开发机类型：个人开发机，输入开发机名称：XTuner微调，选择开发机镜像：Cuda12.2-conda。

步骤3：在镜像详情界面，点击 “使用” 链接，确认使用该镜像。

步骤4：资源配置可以选择 10% （如果有更高资源可以使用，也可以选择更高的资源配置），然后点击 “立即创建” 按钮创建开发机。

步骤5：创建完成后，在开发机列表中可以看到刚创建的开发机，点击 “进入开发机” 链接可以连接进入到开发机。

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825161715.png"/>

当我们准备好开发机之后，就可以进行下一步的微调任务了。

另外，进入开发机之后，请确保自己已经克隆了Tutorial仓库的资料到本地。

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825163109.png"/>

```
mkdir -p /root/InternLM/Tutorial
git clone -b camp3  https://github.com/InternLM/Tutorial /root/InternLM/Tutorial
```



### 2.2 创建虚拟环境



在安装 XTuner 之前，我们需要先创建一个虚拟环境。使用 `Anaconda` 创建一个名为 `xtuner0121` 的虚拟环境，可以直接执行命令。

```
# 创建虚拟环境
conda create -n xtuner0121 python=3.10 -y

# 激活虚拟环境（注意：后续的所有操作都需要在这个虚拟环境中进行）
conda activate xtuner0121

# 安装一些必要的库
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖
pip install transformers==4.39.3
pip install streamlit==1.36.0
```



### 2.3 安装 XTuner



虚拟环境创建完成后，就可以安装 XTuner 了。首先，从 Github 上下载源码。

```
# 创建一个目录，用来存放源代码
mkdir -p /root/InternLM/code

cd /root/InternLM/code

git clone -b v0.1.21  https://github.com/InternLM/XTuner /root/InternLM/code/XTuner
```



其次，进入源码目录，执行安装。

```
# 进入到源码目录
cd /root/InternLM/code/XTuner
conda activate xtuner0121

# 执行安装
pip install -e '.[deepspeed]'
```



> 如果速度太慢可以换成 `pip install -e '.[deepspeed]' -i https://mirrors.aliyun.com/pypi/simple/`

最后，我们可以验证一下安装结果。

```
xtuner version
```

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825170531.png"/>

对于很多初学者而言，我们可能不太熟悉 XTuner 的用法，那么我们可以通过以下命令来查看相关的帮助。

```
xtuner help
```



对于很多的初学者而言，安装好环境意味着成功了一大半！因此我们接下来就可以进入我们的下一步，准备好我们需要的模型、数据集和配置文件，并进行微调训练！

### 2.4 模型准备



软件安装好后，我们就可以准备要微调的模型了。

> 对于学习而言，我们可以使用 InternLM 推出的1.8B的小模型来完成此次微调演示。

对于在 InternStudio 上运行的小伙伴们，可以不用通过 HuggingFace、OpenXLab 或者 Modelscope 进行模型的下载，在开发机中已经为我们提供了模型的本地文件，直接使用就可以了。

> 我们可以通过以下代码一键通过符号链接的方式链接到模型文件，这样既节省了空间，也便于管理。

```
# 创建一个目录，用来存放微调的所有资料，后续的所有操作都在该路径中进行
mkdir -p /root/InternLM/XTuner

cd /root/InternLM/XTuner

mkdir -p Shanghai_AI_Laboratory

ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b Shanghai_AI_Laboratory/internlm2-chat-1_8b
```



执行上述操作后，`Shanghai_AI_Laboratory/internlm2-chat-1_8b` 将直接成为一个符号链接，这个链接指向 `/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b` 的位置。

这意味着，当我们访问 `Shanghai_AI_Laboratory/internlm2-chat-1_8b` 时，实际上就是在访问 `/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b` 目录下的内容。通过这种方式，我们无需复制任何数据，就可以直接利用现有的模型文件进行后续的微调操作，从而节省存储空间并简化文件管理。

模型文件准备好后，我们可以使用`tree`命令来观察目录结构。

```
apt-get install -y tree

tree -l
```

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825170654.png"/>

我们的目录结构应该是这个样子的。

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825170755.png"/>

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">目录结构</summary><div class="snippet-clipboard-content notranslate position-relative overflow-auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; display: flex; justify-content: space-between; margin-bottom: var(--base-size-16); background-color: var(--bgColor-muted, var(--color-canvas-subtle));"><pre class="notranslate" style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: var(--base-size-16); overflow: auto; line-height: 1.45; color: var(--fgColor-default, var(--color-fg-default)); background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-radius: 6px;"><code style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; padding: 0px; margin: 0px; white-space: pre; background: transparent; border-radius: 6px; word-break: normal; border: 0px; display: inline; overflow: visible; line-height: inherit; overflow-wrap: normal;"></code></pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="├── Shanghai_AI_Laboratory
│   └── internlm2-chat-1_8b -> /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b
│       ├── README.md
│       ├── config.json
│       ├── configuration.json
│       ├── configuration_internlm2.py
│       ├── generation_config.json
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── model.safetensors.index.json
│       ├── modeling_internlm2.py
│       ├── special_tokens_map.json
│       ├── tokenization_internlm2.py
│       ├── tokenization_internlm2_fast.py
│       ├── tokenizer.model
│       └── tokenizer_config.json" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: var(--base-text-weight-medium, 500); line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: var(--fgColor-accent, var(--color-accent-fg)); background-color: transparent; box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1) 0s, background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: var(--base-size-8, 8px) !important; width: var(--control-small-size, 28px); height: var(--control-small-size, 28px);"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></clipboard-copy></div></div></details>

> 在目录结构中可以看出，`internlm2-chat-1_8b` 是一个符号链接。

## 3 快速开始



这里我们用 `internlm2-chat-1_8b` 模型，通过 `QLoRA` 的方式来微调一个自己的小助手认知作为案例来进行演示。

### 3.1 微调前的模型对话



我们可以通过网页端的 Demo 来看看微调前 `internlm2-chat-1_8b` 的对话效果。

首先，我们需要准备一个Streamlit程序的脚本。

Streamlit程序的完整代码是：[tools/xtuner_streamlit_demo.py](https://github.com/InternLM/Tutorial/blob/camp3/tools/xtuner_streamlit_demo.py)。

然后，我们可以直接启动应用。

```
conda activate xtuner0121

streamlit run /root/InternLM/Tutorial/tools/xtuner_streamlit_demo.py
```



运行后，在访问前，我们还需要做的就是将端口映射到本地。

然后在本地使用 PowerShell 或者命令行终端，执行以下命令：

> 其中，`8501`是Streamlit程序的服务端口，`43551`需要替换为自己的开发机的端口。

```
ssh -CNg -L 8501:127.0.0.1:8501 root@ssh.intern-ai.org.cn -p 43551
```

然后再输入开发机的root密码。

最后，我们就可以在本地通过浏览器访问：[http://127.0.0.1:8501](http://127.0.0.1:8501/) 来进行对话了。

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825171349.png"/>

### 3.2 指令跟随微调



下面我们对模型进行微调，让模型认识到自己的弟位，了解它自己是你的一个助手。

#### 3.2.1 准数据文件



为了让模型能够认清自己的身份弟位，在询问自己是谁的时候按照我们预期的结果进行回复，我们就需要通过在微调数据集中大量加入这样的数据。我们准备一个数据集文件`datas/assistant.json`，文件内容为对话数据。

```
cd /root/InternLM/XTuner
mkdir -p datas
touch datas/assistant.json
```



为了简化数据文件准备，我们也可以通过脚本生成的方式来准备数据。创建一个脚本文件 `xtuner_generate_assistant.py` ：

```
cd /root/InternLM/XTuner
touch xtuner_generate_assistant.py
```



输入脚本内容并保存：

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">xtuner_generate_assistant.py</summary><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; margin-bottom: var(--base-size-16); display: flex; justify-content: space-between; background-color: var(--bgColor-muted, var(--color-canvas-subtle));"><pre style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: var(--base-size-16); overflow: auto; line-height: 1.45; color: var(--fgColor-default, var(--color-fg-default)); background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-radius: 6px; word-break: normal; min-height: 52px;"><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span></pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="import json

# 设置用户的名字
name = '伍鲜同志'
# 设置需要重复添加的数据次数
n = 8000

# 初始化数据
data = [
    {&quot;conversation&quot;: [{&quot;input&quot;: &quot;请介绍一下你自己&quot;, &quot;output&quot;: &quot;我是{}的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦&quot;.format(name)}]},
    {&quot;conversation&quot;: [{&quot;input&quot;: &quot;你在实战营做什么&quot;, &quot;output&quot;: &quot;我在这里帮助{}完成XTuner微调个人小助手的任务&quot;.format(name)}]}
]

# 通过循环，将初始化的对话数据重复添加到data列表中
for i in range(n):
    data.append(data[0])
    data.append(data[1])

# 将data列表中的数据写入到'datas/assistant.json'文件中
with open('datas/assistant.json', 'w', encoding='utf-8') as f:
    # 使用json.dump方法将数据以JSON格式写入文件
    # ensure_ascii=False 确保中文字符正常显示
    # indent=4 使得文件内容格式化，便于阅读
    json.dump(data, f, ensure_ascii=False, indent=4)
" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: var(--base-text-weight-medium, 500); line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: var(--fgColor-accent, var(--color-accent-fg)); background-color: transparent; box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1) 0s, background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: var(--base-size-8, 8px) !important; width: var(--control-small-size, 28px); height: var(--control-small-size, 28px);"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></clipboard-copy></div></div></details>

> 或者可以直接复制 [tools/xtuner_generate_assistant.py](https://github.com/InternLM/Tutorial/blob/camp3/tools/xtuner_generate_assistant.py)
>
> ```
> cd /root/InternLM/XTuner
> cp /root/InternLM/Tutorial/tools/xtuner_generate_assistant.py ./
> ```

为了训练出自己的小助手，需要将脚本中`name`后面的内容修改为你自己的名称。

```
# 将对应的name进行修改（在第4行的位置）
- name = '伍鲜同志'
+ name = "你自己的名称"
```



> 假如想要让微调后的模型能够完完全全认识到你的身份，我们还可以把第6行的`n`的值调大一点。不过`n`值太大的话容易导致过拟合，无法有效回答其他问题。

然后执行该脚本来生成数据文件。

```
cd /root/InternLM/XTuner
conda activate xtuner0121

python xtuner_generate_assistant.py
```



准备好数据文件后，我们的目录结构应该是这样子的。

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">目录结构</summary><div class="snippet-clipboard-content notranslate position-relative overflow-auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; display: flex; justify-content: space-between; margin-bottom: var(--base-size-16); background-color: var(--bgColor-muted, var(--color-canvas-subtle));"><pre class="notranslate" style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: var(--base-size-16); overflow: auto; line-height: 1.45; color: var(--fgColor-default, var(--color-fg-default)); background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-radius: 6px;"><code style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; padding: 0px; margin: 0px; white-space: pre; background: transparent; border-radius: 6px; word-break: normal; border: 0px; display: inline; overflow: visible; line-height: inherit; overflow-wrap: normal;"></code></pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="├── Shanghai_AI_Laboratory
│   └── internlm2-chat-1_8b -> /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b
│       ├── README.md
│       ├── config.json
│       ├── configuration.json
│       ├── configuration_internlm2.py
│       ├── generation_config.json
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── model.safetensors.index.json
│       ├── modeling_internlm2.py
│       ├── special_tokens_map.json
│       ├── tokenization_internlm2.py
│       ├── tokenization_internlm2_fast.py
│       ├── tokenizer.model
│       └── tokenizer_config.json
├── datas
│   └── assistant.json
├── xtuner_generate_assistant.py" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: var(--base-text-weight-medium, 500); line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: var(--fgColor-accent, var(--color-accent-fg)); background-color: transparent; box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1) 0s, background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: var(--base-size-8, 8px) !important; width: var(--control-small-size, 28px); height: var(--control-small-size, 28px);"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></clipboard-copy></div></div></details>

#### 3.2.2 准备配置文件



在准备好了模型和数据集后，我们就要根据我们选择的微调方法结合微调方案来找到与我们最匹配的配置文件了，从而减少我们对配置文件的修改量。

> 配置文件其实是一种用于定义和控制模型训练和测试过程中各个方面的参数和设置的工具。

##### 3.2.2.1 列出支持的配置文件



XTuner 提供多个开箱即用的配置文件，可以通过以下命令查看。

> `xtuner list-cfg` 命令用于列出内置的所有配置文件。参数 `-p` 或 `--pattern` 表示模式匹配，后面跟着的内容将会在所有的配置文件里进行模糊匹配搜索，然后返回最有可能得内容。比如我们这里微调的是书生·浦语的模型，我们就可以匹配搜索 `internlm2`。

```
conda activate xtuner0121

xtuner list-cfg -p internlm2
```



<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">配置文件名的解释</summary><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);"></strong><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);"></strong></p><markdown-accessiblity-table data-catalyst="" style="box-sizing: border-box; display: block;"><table style="box-sizing: border-box; border-spacing: 0px; border-collapse: collapse; margin-top: 0px; margin-bottom: var(--base-size-16); display: block; width: max-content; max-width: 100%; overflow: auto;"><thead style="box-sizing: border-box;"><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><th style="box-sizing: border-box; padding: 6px 13px; font-weight: var(--base-text-weight-semibold, 600); border: 1px solid var(--borderColor-default, var(--color-border-default));"></th><th style="box-sizing: border-box; padding: 6px 13px; font-weight: var(--base-text-weight-semibold, 600); border: 1px solid var(--borderColor-default, var(--color-border-default));"></th><th style="box-sizing: border-box; padding: 6px 13px; font-weight: var(--base-text-weight-semibold, 600); border: 1px solid var(--borderColor-default, var(--color-border-default));"></th></tr></thead><tbody style="box-sizing: border-box;"><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr></tbody></table></markdown-accessiblity-table></details>

##### 3.2.2.2 复制一个预设的配置文件



由于我们是对`internlm2-chat-1_8b`模型进行指令微调，所以与我们的需求最匹配的配置文件是 `internlm2_chat_1_8b_qlora_alpaca_e3`，这里就复制该配置文件。

> `xtuner copy-cfg` 命令用于复制一个内置的配置文件。该命令需要两个参数：`CONFIG` 代表需要复制的配置文件名称，`SAVE_PATH` 代表复制的目标路径。在我们的输入的这个命令中，我们的 `CONFIG` 对应的是上面搜索到的 `internlm2_chat_1_8b_qlora_alpaca_e3` ,而 `SAVE_PATH` 则是当前目录 `.`。

```
cd /root/InternLM/XTuner
conda activate xtuner0121

xtuner copy-cfg internlm2_chat_1_8b_qlora_alpaca_e3 .
```



复制好配置文件后，我们的目录结构应该是这样子的。

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825171957.png"/>

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">目录结构</summary><div class="snippet-clipboard-content notranslate position-relative overflow-auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; display: flex; justify-content: space-between; margin-bottom: var(--base-size-16); background-color: var(--bgColor-muted, var(--color-canvas-subtle));"><pre class="notranslate" style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: var(--base-size-16); overflow: auto; line-height: 1.45; color: var(--fgColor-default, var(--color-fg-default)); background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-radius: 6px;"><code style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; padding: 0px; margin: 0px; white-space: pre; background: transparent; border-radius: 6px; word-break: normal; border: 0px; display: inline; overflow: visible; line-height: inherit; overflow-wrap: normal;"></code></pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="├── Shanghai_AI_Laboratory
│   └── internlm2-chat-1_8b -> /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b
│       ├── README.md
│       ├── config.json
│       ├── configuration.json
│       ├── configuration_internlm2.py
│       ├── generation_config.json
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── model.safetensors.index.json
│       ├── modeling_internlm2.py
│       ├── special_tokens_map.json
│       ├── tokenization_internlm2.py
│       ├── tokenization_internlm2_fast.py
│       ├── tokenizer.model
│       └── tokenizer_config.json
├── datas
│   └── assistant.json
├── internlm2_chat_1_8b_qlora_alpaca_e3_copy.py
├── xtuner_generate_assistant.py" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: var(--base-text-weight-medium, 500); line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: var(--fgColor-accent, var(--color-accent-fg)); background-color: transparent; box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1) 0s, background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: var(--base-size-8, 8px) !important; width: var(--control-small-size, 28px); height: var(--control-small-size, 28px);"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></clipboard-copy></div></div></details>

##### 3.2.2.3 对配置文件进行修改



在选择了一个最匹配的配置文件并准备好其他内容后，下面我们要做的事情就是根据我们自己的内容对该配置文件进行调整，使其能够满足我们实际训练的要求。

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;"><b style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);">配置文件介绍</b></summary><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"></p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);"></strong></p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);"></strong></p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);"></strong></p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);"></strong></p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);"></strong></p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"></p></details>

下面我们将根据项目的需求一步步的进行修改和调整吧！

在 PART 1 的部分，由于我们不再需要在 HuggingFace 上自动下载模型，因此我们先要更换模型的路径以及数据集的路径为我们本地的路径。

为了训练过程中能够实时观察到模型的变化情况，XTuner 贴心的推出了一个 `evaluation_inputs` 的参数来让我们能够设置多个问题来确保模型在训练过程中的变化是朝着我们想要的方向前进的。我们可以添加自己的输入。

在 PART 3 的部分，由于我们准备的数据集是 JSON 格式的数据，并且对话内容已经是 `input` 和 `output` 的数据对，所以不需要进行格式转换。

```
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b'
+ pretrained_model_name_or_path = '/root/InternLM/XTuner/Shanghai_AI_Laboratory/internlm2-chat-1_8b'

- alpaca_en_path = 'tatsu-lab/alpaca'
+ alpaca_en_path = 'datas/assistant.json'

evaluation_inputs = [
-    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
+    '请介绍一下你自己', 'Please introduce yourself'
]

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=alpaca_en_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)
```



除此之外，我们还可以对一些重要的参数进行调整，包括学习率（lr）、训练的轮数（max_epochs）等等。

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">常用参数介绍</summary><markdown-accessiblity-table data-catalyst="" style="box-sizing: border-box; display: block;"><table style="box-sizing: border-box; border-spacing: 0px; border-collapse: collapse; margin-top: 0px; margin-bottom: var(--base-size-16); display: block; width: max-content; max-width: 100%; overflow: auto;"><thead style="box-sizing: border-box;"><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><th style="box-sizing: border-box; padding: 6px 13px; font-weight: var(--base-text-weight-semibold, 600); border: 1px solid var(--borderColor-default, var(--color-border-default));"></th><th style="box-sizing: border-box; padding: 6px 13px; font-weight: var(--base-text-weight-semibold, 600); border: 1px solid var(--borderColor-default, var(--color-border-default));"></th></tr></thead><tbody style="box-sizing: border-box;"><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600); margin-bottom: 0px;"></strong></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr></tbody></table></markdown-accessiblity-table><blockquote style="box-sizing: border-box; margin-top: 0px; margin-right: 0px; margin-bottom: var(--base-size-16); margin-left: 0px; padding: 0px 1em; color: var(--fgColor-muted, var(--color-fg-muted)); border-left: .25em solid var(--borderColor-default, var(--color-border-default));"><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: 0px;"><code style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: var(--bgColor-neutral-muted, var(--color-neutral-muted)); border-radius: 6px;"></code><code style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; padding: 0.2em 0.4em; margin: 0px; white-space: break-spaces; background-color: var(--bgColor-neutral-muted, var(--color-neutral-muted)); border-radius: 6px;"></code></p></blockquote></details>

修改完后的完整的配置文件是：[configs/internlm2_chat_1_8b_qlora_alpaca_e3_copy.py](https://github.com/InternLM/Tutorial/blob/camp3/configs/internlm2_chat_1_8b_qlora_alpaca_e3_copy.py)。

> 可以直接复制到当前目录。
>
> ```
> cd /root/InternLM/XTuner
> cp /root/InternLM/Tutorial/configs/internlm2_chat_1_8b_qlora_alpaca_e3_copy.py ./
> ```

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">internlm2_chat_1_8b_qlora_alpaca_e3_copy.py</summary><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; margin-bottom: var(--base-size-16); display: flex; justify-content: space-between; background-color: var(--bgColor-muted, var(--color-canvas-subtle));"><pre style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: var(--base-size-16); overflow: auto; line-height: 1.45; color: var(--fgColor-default, var(--color-fg-default)); background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-radius: 6px; word-break: normal; min-height: 52px;"><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-k" style="box-sizing: border-box; color: var(--color-prettylights-syntax-keyword);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-v" style="box-sizing: border-box; color: var(--color-prettylights-syntax-variable);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s" style="box-sizing: border-box; color: var(--color-prettylights-syntax-string);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c" style="box-sizing: border-box; color: var(--color-prettylights-syntax-comment);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-en" style="box-sizing: border-box; color: var(--color-prettylights-syntax-entity);"></span><span class="pl-s1" style="box-sizing: border-box;"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span><span class="pl-c1" style="box-sizing: border-box; color: var(--color-prettylights-syntax-constant);"></span></pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/root/InternLM/XTuner/Shanghai_AI_Laboratory/internlm2-chat-1_8b'
use_varlen_attn = False

# Data
alpaca_en_path = 'datas/assistant.json'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 2048
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler &amp; Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = SYSTEM_TEMPLATE.alpaca
evaluation_inputs = [
    '请介绍一下你自己', 'Please introduce yourself'
]

#######################################################################
#                      PART 2  Model &amp; Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))

#######################################################################
#                      PART 3  Dataset &amp; Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=alpaca_en,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Scheduler &amp; Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: var(--base-text-weight-medium, 500); line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: var(--fgColor-accent, var(--color-accent-fg)); background-color: transparent; box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1) 0s, background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: var(--base-size-8, 8px) !important; width: var(--control-small-size, 28px); height: var(--control-small-size, 28px);"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></clipboard-copy></div></div></details>

#### 3.2.3 启动微调



完成了所有的准备工作后，我们就可以正式的开始我们下一阶段的旅程：XTuner 启动~！

当我们准备好了所有内容，我们只需要将使用 `xtuner train` 命令令即可开始训练。

> `xtuner train` 命令用于启动模型微调进程。该命令需要一个参数：`CONFIG` 用于指定微调配置文件。这里我们使用修改好的配置文件 `internlm2_chat_1_8b_qlora_alpaca_e3_copy.py`。
> 训练过程中产生的所有文件，包括日志、配置文件、检查点文件、微调后的模型等，默认保存在 `work_dirs` 目录下，我们也可以通过添加 `--work-dir` 指定特定的文件保存位置。

```
cd /root/InternLM/XTuner
conda activate xtuner0121

xtuner train ./internlm2_chat_1_8b_qlora_alpaca_e3_copy.py
```



在训练完后，我们的目录结构应该是这样子的。

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825184706.png"/>

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">目录结构</summary><div class="snippet-clipboard-content notranslate position-relative overflow-auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; display: flex; justify-content: space-between; margin-bottom: var(--base-size-16); background-color: var(--bgColor-muted, var(--color-canvas-subtle));"><pre class="notranslate" style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: var(--base-size-16); overflow: auto; line-height: 1.45; color: var(--fgColor-default, var(--color-fg-default)); background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-radius: 6px;"><code style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; padding: 0px; margin: 0px; white-space: pre; background: transparent; border-radius: 6px; word-break: normal; border: 0px; display: inline; overflow: visible; line-height: inherit; overflow-wrap: normal;"></code></pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="├── work_dirs
│   └── internlm2_chat_1_8b_qlora_alpaca_e3_copy
│       ├── 20240626_222727
│       │   ├── 20240626_222727.log
│       │   └── vis_data
│       │       ├── 20240626_222727.json
│       │       ├── config.py
│       │       ├── eval_outputs_iter_95.txt
│       │       └── scalars.json
│       ├── internlm2_chat_1_8b_qlora_alpaca_e3_copy.py
│       ├── iter_96.pth
│       └── last_checkpoint" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: var(--base-text-weight-medium, 500); line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: var(--fgColor-accent, var(--color-accent-fg)); background-color: transparent; box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1) 0s, background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: var(--base-size-8, 8px) !important; width: var(--control-small-size, 28px); height: var(--control-small-size, 28px);"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></clipboard-copy></div></div></details>

#### 3.2.4 模型格式转换



模型转换的本质其实就是将原本使用 Pytorch 训练出来的模型权重文件转换为目前通用的 HuggingFace 格式文件，那么我们可以通过以下命令来实现一键转换。

我们可以使用 `xtuner convert pth_to_hf` 命令来进行模型格式转换。

> `xtuner convert pth_to_hf` 命令用于进行模型格式转换。该命令需要三个参数：`CONFIG` 表示微调的配置文件， `PATH_TO_PTH_MODEL` 表示微调的模型权重文件路径，即要转换的模型权重， `SAVE_PATH_TO_HF_MODEL` 表示转换后的 HuggingFace 格式文件的保存路径。

除此之外，我们其实还可以在转换的命令中添加几个额外的参数，包括：

| 参数名                | 解释                                         |
| --------------------- | -------------------------------------------- |
| --fp32                | 代表以fp32的精度开启，假如不输入则默认为fp16 |
| --max-shard-size {GB} | 代表每个权重文件最大的大小（默认为2GB）      |

```
cd /root/InternLM/XTuner
conda activate xtuner0121

# 先获取最后保存的一个pth文件
pth_file=`ls -t ./work_dirs/internlm2_chat_1_8b_qlora_alpaca_e3_copy/*.pth | head -n 1`
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm2_chat_1_8b_qlora_alpaca_e3_copy.py ${pth_file} ./hf
```



模型格式转换完成后，我们的目录结构应该是这样子的。

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825185315.png"/>

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">目录结构</summary><div class="snippet-clipboard-content notranslate position-relative overflow-auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; display: flex; justify-content: space-between; margin-bottom: var(--base-size-16); background-color: var(--bgColor-muted, var(--color-canvas-subtle));"><pre class="notranslate" style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: var(--base-size-16); overflow: auto; line-height: 1.45; color: var(--fgColor-default, var(--color-fg-default)); background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-radius: 6px;"><code style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; padding: 0px; margin: 0px; white-space: pre; background: transparent; border-radius: 6px; word-break: normal; border: 0px; display: inline; overflow: visible; line-height: inherit; overflow-wrap: normal;"></code></pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="├── hf
│   ├── README.md
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── xtuner_config.py" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: var(--base-text-weight-medium, 500); line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: var(--fgColor-accent, var(--color-accent-fg)); background-color: transparent; box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1) 0s, background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: var(--base-size-8, 8px) !important; width: var(--control-small-size, 28px); height: var(--control-small-size, 28px);"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></clipboard-copy></div></div></details>

转换完成后，可以看到模型被转换为 HuggingFace 中常用的 .bin 格式文件，这就代表着文件成功被转化为 HuggingFace 格式了。

此时，hf 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”

> 可以简单理解：LoRA 模型文件 = Adapter

#### 3.2.5 模型合并



对于 LoRA 或者 QLoRA 微调出来的模型其实并不是一个完整的模型，而是一个额外的层（Adapter），训练完的这个层最终还是要与原模型进行合并才能被正常的使用。

> 对于全量微调的模型（full）其实是不需要进行整合这一步的，因为全量微调修改的是原模型的权重而非微调一个新的 Adapter ，因此是不需要进行模型整合的。

在 XTuner 中提供了一键合并的命令 `xtuner convert merge`，在使用前我们需要准备好三个路径，包括原模型的路径、训练好的 Adapter 层的（模型格式转换后的）路径以及最终保存的路径。

> `xtuner convert merge`命令用于合并模型。该命令需要三个参数：`LLM` 表示原模型路径，`ADAPTER` 表示 Adapter 层的路径， `SAVE_PATH` 表示合并后的模型最终的保存路径。

在模型合并这一步还有其他很多的可选参数，包括：

| 参数名                 | 解释                                                         |
| ---------------------- | ------------------------------------------------------------ |
| --max-shard-size {GB}  | 代表每个权重文件最大的大小（默认为2GB）                      |
| --device {device_name} | 这里指的就是device的名称，可选择的有cuda、cpu和auto，默认为cuda即使用gpu进行运算 |
| --is-clip              | 这个参数主要用于确定模型是不是CLIP模型，假如是的话就要加上，不是就不需要添加 |

```
cd /root/InternLM/XTuner
conda activate xtuner0121

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert merge /root/InternLM/XTuner/Shanghai_AI_Laboratory/internlm2-chat-1_8b ./hf ./merged --max-shard-size 2GB
```



模型合并完成后，我们的目录结构应该是这样子的。

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825185753.png"/>

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">目录结构</summary><div class="snippet-clipboard-content notranslate position-relative overflow-auto" style="box-sizing: border-box; position: relative !important; overflow: auto !important; display: flex; justify-content: space-between; margin-bottom: var(--base-size-16); background-color: var(--bgColor-muted, var(--color-canvas-subtle));"><pre class="notranslate" style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; margin-top: 0px; margin-bottom: 0px; overflow-wrap: normal; padding: var(--base-size-16); overflow: auto; line-height: 1.45; color: var(--fgColor-default, var(--color-fg-default)); background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-radius: 6px;"><code style="box-sizing: border-box; font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace); font-size: 13.6px; padding: 0px; margin: 0px; white-space: pre; background: transparent; border-radius: 6px; word-break: normal; border: 0px; display: inline; overflow: visible; line-height: inherit; overflow-wrap: normal;"></code></pre><div class="zeroclipboard-container" style="box-sizing: border-box; display: block; animation: auto ease 0s 1 normal none running none;"><clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="├── merged
│   ├── config.json
│   ├── configuration_internlm2.py
│   ├── generation_config.json
│   ├── modeling_internlm2.py
│   ├── pytorch_model-00001-of-00002.bin
│   ├── pytorch_model-00002-of-00002.bin
│   ├── pytorch_model.bin.index.json
│   ├── special_tokens_map.json
│   ├── tokenization_internlm2.py
│   ├── tokenization_internlm2_fast.py
│   ├── tokenizer.json
│   ├── tokenizer.model
│   └── tokenizer_config.json" tabindex="0" role="button" style="box-sizing: border-box; position: relative; display: flex !important; padding: 0px !important; font-size: 14px; font-weight: var(--base-text-weight-medium, 500); line-height: 20px; white-space: nowrap; vertical-align: middle; cursor: pointer; user-select: none; border: 0px; border-radius: 6px; appearance: none; color: var(--fgColor-accent, var(--color-accent-fg)); background-color: transparent; box-shadow: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1) 0s, background-color, box-shadow, border-color; justify-content: center !important; align-items: center !important; margin: var(--base-size-8, 8px) !important; width: var(--control-small-size, 28px); height: var(--control-small-size, 28px);"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></clipboard-copy></div></div></details>

在模型合并完成后，我们就可以看到最终的模型和原模型文件夹非常相似，包括了分词器、权重文件、配置信息等等。

### 3.3 微调后的模型对话



微调完成后，我们可以再次运行`xtuner_streamlit_demo.py`脚本来观察微调后的对话效果，不过在运行之前，我们需要将脚本中的模型路径修改为微调后的模型的路径。

```
# 直接修改脚本文件第18行
- model_name_or_path = "/root/InternLM/XTuner/Shanghai_AI_Laboratory/internlm2-chat-1_8b"
+ model_name_or_path = "/root/InternLM/XTuner/merged"
```



然后，我们可以直接启动应用。

```
conda activate xtuner0121

streamlit run /root/InternLM/Tutorial/tools/xtuner_streamlit_demo.py
```



运行后，确保端口映射正常，如果映射已断开则需要重新做一次端口映射。

```
ssh -CNg -L 8501:127.0.0.1:8501 root@ssh.intern-ai.org.cn -p 43551
```



最后，通过浏览器访问：[http://127.0.0.1:8501](http://127.0.0.1:8501/) 来进行对话了。

<img src="https://raw.githubusercontent.com/Chl681006/photo_save/main/20240825191836.png"/>

## 4 小结



经过本节的学习，带领着大家跑通了 XTuner 的完整流程，我们学会了指令跟随微调，但是我的助手并灭有变成我自己的。

当我们在测试完模型认为其满足我们的需求后，就可以对模型进行量化部署等操作了，这部分的内容在之后关于 LMDeploy 的课程中将会详细的进行讲解，敬请期待后续的课程吧！

关于XTuner的更多高级进阶知识，请访问[XTuner微调高级进阶](https://github.com/InternLM/Tutorial/blob/camp3/docs/L1/XTuner/xtuner_finetune_advance.md)。