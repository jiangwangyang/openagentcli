use anyhow::{anyhow, bail, Result};
use futures::stream::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_stream::wrappers::LinesStream;
use tokio_util::io::StreamReader;

// ---------- 数据结构定义 ----------
#[derive(Debug, Clone, Serialize)]
struct Message {
    role: String,
    content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

#[derive(Debug, Clone)]
enum ContentBlockBuilder {
    Thinking {
        thinking: String,
        signature: String,
    },
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
}

impl ContentBlockBuilder {
    fn into_block(self) -> Result<ContentBlock> {
        match self {
            ContentBlockBuilder::Thinking {
                thinking,
                signature,
            } => Ok(ContentBlock::Thinking {
                thinking,
                signature,
            }),
            ContentBlockBuilder::Text { text } => Ok(ContentBlock::Text { text }),
            ContentBlockBuilder::ToolUse { id, name, input } => {
                let parsed = if input.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::from_str(&input)
                        .map_err(|e| anyhow!("解析工具输入 JSON 失败: {}", e))?
                };
                Ok(ContentBlock::ToolUse {
                    id,
                    name,
                    input: parsed,
                })
            }
        }
    }
}

// ---------- Anthropic 流式事件解析 ----------
#[derive(Debug, Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    // index: Option<u32>,
    content_block: Option<ContentBlockStart>,
    delta: Option<Delta>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStart {
    #[serde(rename = "type")]
    block_type: String,
    id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum Delta {
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

// 工具调用方法
async fn execute_tool(
    name: &str,
    input: &serde_json::Value,
    _work_dir: &str,
) -> Result<(String, bool)> {
    // 1. 检查工具名
    if name != "command" {
        return Ok((format!("Unknown tool: {}", name), true));
    }

    // 2. 解析 args 数组
    let args = match input.get("args").and_then(|v| v.as_array()) {
        Some(arr) => {
            let mut vec = Vec::new();
            for val in arr {
                if let Some(s) = val.as_str() {
                    vec.push(s.to_string());
                } else {
                    return Ok((format!("Argument must be a string: {}", val), true));
                }
            }
            vec
        }
        None => return Ok(("Missing 'args' array or not an array".to_string(), true)),
    };

    if args.is_empty() {
        return Ok(("No command provided".to_string(), true));
    }

    // 3. 分离命令和参数
    let cmd = &args[0];
    let cmd_args = &args[1..];

    // 4. 使用 tokio::process::Command 异步执行
    let output = tokio::process::Command::new(cmd)
        .args(cmd_args)
        .current_dir(_work_dir)
        .output()
        .await;

    // 5. 处理执行结果
    match output {
        Ok(out) => {
            if out.status.success() {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                Ok((stdout, false))
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                let msg = format!(
                    "Command failed with exit code {:?}: {}",
                    out.status.code(),
                    stderr
                );
                Ok((msg, true))
            }
        }
        Err(e) => {
            let msg = format!("Failed to execute command: {}", e);
            Ok((msg, true))
        }
    }
}

// 发送消息请求并返回流式事件迭代器（异步流）
async fn messages_stream(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
) -> Result<Vec<ContentBlock>> {
    // 1. 构造请求
    let url = format!("{}/messages", base_url);
    let mut body = json!({
        "messages": messages,
        "system": system_prompt,
        "model": model,
        "max_tokens": 1 << 14,
        "stream": true,
    });
    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }

    // 2. 发送请求
    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let error_text = resp.text().await?;
        bail!("Anthropic API 错误: {}", error_text);
    }

    // 3. 将响应体转换为逐行流，并解析 SSE 事件
    let stream = resp
        .bytes_stream()
        .map(|result| result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
    let reader = BufReader::new(StreamReader::new(stream));
    let lines = LinesStream::new(reader.lines());
    let event_stream = lines.filter_map(|line| async {
        match line {
            Ok(line) if line.starts_with("data: ") => {
                let data = line[5..].trim();
                match serde_json::from_str::<StreamEvent>(data) {
                    Ok(event) => Some(Ok(event)),
                    Err(_) => None,
                }
            }
            Ok(_) => None,
            Err(e) => Some(Err(anyhow!("读取行失败: {}", e))),
        }
    });
    futures::pin_mut!(event_stream);

    // 4. 处理事件流
    let mut model_blocks_builders: Vec<ContentBlockBuilder> = Vec::new();
    while let Some(event_res) = event_stream.next().await {
        let event = event_res?;
        match event.event_type.as_str() {
            "content_block_start" => {
                if let Some(block_start) = event.content_block {
                    match block_start.block_type.as_str() {
                        "thinking" => {
                            model_blocks_builders.push(ContentBlockBuilder::Thinking {
                                thinking: String::new(),
                                signature: String::new(),
                            });
                        }
                        "text" => {
                            model_blocks_builders.push(ContentBlockBuilder::Text {
                                text: String::new(),
                            });
                        }
                        "tool_use" => {
                            let id = block_start.id.ok_or_else(|| anyhow!("tool_use 缺少 id"))?;
                            let name = block_start
                                .name
                                .ok_or_else(|| anyhow!("tool_use 缺少 name"))?;
                            model_blocks_builders.push(ContentBlockBuilder::ToolUse {
                                id,
                                name,
                                input: String::new(),
                            });
                        }
                        _ => {
                            // 忽略未知类型
                        }
                    }
                }
            }
            "content_block_delta" => {
                if let Some(delta) = event.delta {
                    let last = model_blocks_builders
                        .last_mut()
                        .ok_or_else(|| anyhow!("收到 delta 但无对应块"))?;
                    match delta {
                        Delta::ThinkingDelta {
                            thinking: delta_text,
                        } => {
                            if let ContentBlockBuilder::Thinking { thinking, .. } = last {
                                thinking.push_str(&delta_text);
                            } else {
                                bail!("delta 类型与当前块不匹配");
                            }
                        }
                        Delta::SignatureDelta {
                            signature: delta_text,
                        } => {
                            if let ContentBlockBuilder::Thinking { signature, .. } = last {
                                signature.push_str(&delta_text);
                            } else {
                                bail!("delta 类型与当前块不匹配");
                            }
                        }
                        Delta::TextDelta { text: delta_text } => {
                            if let ContentBlockBuilder::Text { text } = last {
                                text.push_str(&delta_text);
                            } else {
                                bail!("delta 类型与当前块不匹配");
                            }
                        }
                        Delta::InputJsonDelta { partial_json } => {
                            if let ContentBlockBuilder::ToolUse { input, .. } = last {
                                input.push_str(&partial_json);
                            } else {
                                bail!("delta 类型与当前块不匹配");
                            }
                        }
                    }
                }
            }
            _ => {
                // 忽略其他事件
            }
        }
    }

    // 5. 返回消息结果
    let assistant_blocks: Vec<ContentBlock> = model_blocks_builders
        .into_iter()
        .filter_map(|builder| builder.into_block().ok())
        .collect();

    Ok(assistant_blocks)
}

// ---------- main ----------
#[tokio::main]
async fn main() -> Result<()> {
    // 读取输入 args[0] 是程序名，所以真正输入从索引 1 开始
    let mut args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "run" {
        args.remove(1);
    }
    let inputs = args.into_iter().skip(1).collect::<Vec<_>>();
    let prompt = inputs.join(" ");
    println!("{:?}", prompt);

    // 创建http客户端
    let client = Client::new();

    // 读取配置
    let base_url = env::var("OPENAGENT_BASE_URL").unwrap_or("".to_string());
    let api_key = env::var("OPENAGENT_API_KEY").unwrap_or("".to_string());
    let model = env::var("OPENAGENT_MODEL").unwrap_or("".to_string());

    // 初始消息
    let mut messages = vec![Message {
        role: "user".to_string(),
        content: vec![ContentBlock::Text { text: prompt }],
    }];

    // 工具定义
    let tools = vec![json!({
        "name": "command",
        "description": "Execute commands on your system",
        "input_schema": {
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Argument list."
                }
            },
            "required": ["args"]
        }
    })];

    // 系统提示
    let system_prompt = tokio::fs::read_to_string("AGENTS.md")
        .await
        .unwrap_or_default();

    // 当前目录
    let cwd = env::current_dir()?.to_str().unwrap().to_string();

    // 执行 Agent
    loop {
        // 1. 发送请求，获取模型消息
        let assistant_blocks = messages_stream(
            &client,
            &base_url,
            &api_key,
            &model,
            &system_prompt,
            &messages,
            &tools,
        )
        .await?;

        // 2. 将模型返回添加到消息列表
        messages.push(Message {
            role: "assistant".to_string(),
            content: assistant_blocks,
        });

        // 提取出工具调用消息
        let tool_uses: Vec<&ContentBlock> = messages
            .last() // 或 .iter_mut() 等
            .unwrap()
            .content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
            .collect();

        // 3. 如果没有工具调用，结束循环
        if tool_uses.is_empty() {
            break;
        }

        // 4. 执行工具调用
        let mut tool_results = Vec::new();
        for block in tool_uses {
            if let ContentBlock::ToolUse { id, name, input } = block {
                let (content, is_error) = execute_tool(name, input, &cwd).await?;
                tool_results.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content,
                    is_error,
                });
            }
        }

        // 5. 将工具结果作为用户消息追加
        messages.push(Message {
            role: "user".to_string(),
            content: tool_results,
        });
    }

    println!("{:#?}", messages);
    Ok(())
}
