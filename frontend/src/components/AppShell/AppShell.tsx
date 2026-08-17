import { SafetyCertificateOutlined } from "@ant-design/icons";
import { Layout, Tag } from "antd";
import { useEffect } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { useAppStore } from "../../stores/appStore";
import { NexusMark } from "../NexusMark/NexusMark";

const navItems = [
  { key: "/", label: "协同工作台", english: "Workspace" },
  { key: "/knowledge", label: "企业知识", english: "Knowledge" },
  { key: "/monitor", label: "运行观测", english: "Observability" },
];

export function AppShell() {
  const location = useLocation();
  const { health, healthError, healthLoading, checkHealth } = useAppStore();

  useEffect(() => {
    void checkHealth();
    const timer = window.setInterval(() => void checkHealth(), 30_000);
    return () => window.clearInterval(timer);
  }, [checkHealth]);

  const online = health?.status === "ok";
  return (
    <Layout className="app-shell">
      <header className="topbar">
        <Link to="/" className="brand-lockup" aria-label="NexusOps 协同工作台">
          <NexusMark />
          <span className="brand-copy"><strong>NexusOps</strong><small>企业智能运营协同中枢</small></span>
        </Link>
        <nav className="top-nav" aria-label="一级导航">
          {navItems.map((item) => (
            <Link className={location.pathname === item.key ? "active" : ""} key={item.key} to={item.key}>
              <span>{item.label}</span><small>{item.english}</small>
            </Link>
          ))}
        </nav>
        <Tag className={`health-indicator ${online ? "online" : "offline"}`} data-testid="health-indicator">
          <SafetyCertificateOutlined /> {healthLoading ? "正在检查" : online ? "系统运行正常" : "需要关注"}
        </Tag>
        {healthError && <span className="health-error" title={healthError}>服务暂不可用</span>}
      </header>
      <Layout className="content-layout">
        <main className="page-content"><Outlet /></main>
      </Layout>
    </Layout>
  );
}
