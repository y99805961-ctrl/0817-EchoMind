import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { ConfigProvider } from "antd";
import "antd/dist/reset.css";
import "./styles/tokens.css";
import "./styles/global.css";
import { App } from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: "#4e8665",
          colorInfo: "#79afc5",
          colorSuccess: "#4e8665",
          colorWarning: "#e1a258",
          colorError: "#c76f69",
          colorText: "#27332d",
          colorTextSecondary: "#66726b",
          colorBgBase: "#f8faf6",
          colorBorder: "#dfe8e1",
          borderRadius: 12,
          fontFamily: 'Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
        },
      }}
    >
      <BrowserRouter><App /></BrowserRouter>
    </ConfigProvider>
  </React.StrictMode>,
);
