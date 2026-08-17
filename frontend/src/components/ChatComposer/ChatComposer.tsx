import { ArrowUpOutlined, MacCommandOutlined } from "@ant-design/icons";
import { Button, Input } from "antd";
import { useState } from "react";

interface ChatComposerProps {
  disabled?: boolean;
  onSend: (value: string) => void;
  initialValue?: string;
}

export function ChatComposer({ disabled = false, onSend, initialValue = "" }: ChatComposerProps) {
  const [value, setValue] = useState(initialValue);

  function submit() {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
  }

  return (
    <div className="composer-wrap">
      <MacCommandOutlined className="composer-icon" aria-hidden="true" />
      <Input.TextArea
        data-testid="chat-composer"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
        autoSize={{ minRows: 1, maxRows: 5 }}
        placeholder="描述需要处理的业务请求…"
        disabled={disabled}
        aria-label="描述需要处理的业务请求"
      />
      <Button
        data-testid="send-button"
        className="send-button"
        type="primary"
        shape="circle"
        icon={<ArrowUpOutlined />}
        onClick={submit}
        disabled={disabled || !value.trim()}
        aria-label="发送消息"
      />
    </div>
  );
}
